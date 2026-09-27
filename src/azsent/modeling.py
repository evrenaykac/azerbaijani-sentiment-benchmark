"""Model: encoder + polarity/morphology injection + pooling + heads.

Injection points (Reviewer 1, item 5):
  post : conditioning is applied to the *final* contextual token
         representations, before mean pooling (the configuration reported in
         the original submission - i.e. AFTER the transformer stack).
  mid  : conditioning is applied to the hidden states entering encoder layer
         `mid_layer` via a forward pre-hook (i.e. WITHIN the encoder). Reported
         as an additional ablation in the resubmission.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import D_FEAT


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb: float):
    return GradReverse.apply(x, lamb)


class Injector(nn.Module):
    def __init__(self, hidden: int, mode: str = "film", d_feat: int = D_FEAT):
        super().__init__()
        self.mode = mode
        if mode == "additive":
            self.proj = nn.Linear(d_feat, hidden, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)
        elif mode == "film":
            self.film = nn.Linear(d_feat, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            with torch.no_grad():
                self.film.bias.zero_()
                self.film.bias[: hidden].fill_(1.0)  # gamma starts at 1, beta at 0
        elif mode != "none":
            raise ValueError(mode)

    def forward(self, h: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return h
        if self.mode == "additive":
            return h + self.proj(feats)
        gb = self.film(feats)
        H = h.shape[-1]
        gamma, beta = gb[..., :H], gb[..., H:]
        return gamma * h + beta


def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


class SentModel(nn.Module):
    def __init__(
        self,
        backbone_id: str,
        inject: str = "none",
        inject_point: str = "post",
        mid_layer: int = 6,
        dann: bool = False,
        n_domains: int = 5,
        cache_dir: str | None = None,
        local_dir: str | None = None,
    ):
        super().__init__()
        from transformers import AutoModel

        src = local_dir or backbone_id
        self.encoder = AutoModel.from_pretrained(src, cache_dir=cache_dir).float()
        hidden = self.encoder.config.hidden_size
        self.hidden = hidden
        self.inject_mode = inject
        self.inject_point = inject_point if inject != "none" else "post"
        self.mid_layer = mid_layer
        self.injector = Injector(hidden, inject) if inject != "none" else None
        self.classifier = nn.Linear(hidden, 3)
        self.dann = dann
        if dann:
            self.domain_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_domains)
            )
        self._hook_handle = None
        self._pending = None  # feats tensor consumed by the mid hook
        if self.injector is not None and self.inject_point == "mid":
            self._register_mid_hook()

    # --- mid-injection machinery ---------------------------------------------
    def _encoder_layers(self):
        enc = self.encoder
        for attr in ("encoder",):
            if hasattr(enc, attr) and hasattr(getattr(enc, attr), "layer"):
                return getattr(enc, attr).layer
        raise RuntimeError("Cannot locate encoder.layer list for mid injection on this architecture")

    def _register_mid_hook(self):
        layers = self._encoder_layers()
        k = min(self.mid_layer, len(layers) - 1)

        def pre_hook(module, args, kwargs):
            if self._pending is None:
                return None
            hidden_states = args[0]
            feats = self._pending
            if feats.shape[1] != hidden_states.shape[1]:  # safety on truncation mismatch
                T = min(feats.shape[1], hidden_states.shape[1])
                feats = feats[:, :T]
                pad = hidden_states.shape[1] - T
                if pad > 0:
                    feats = F.pad(feats, (0, 0, 0, pad))
            new_h = self.injector(hidden_states, feats)
            return (new_h,) + tuple(args[1:]), kwargs

        self._hook_handle = layers[k].register_forward_pre_hook(pre_hook, with_kwargs=True)

    def forward(self, input_ids, attention_mask, feats=None, token_type_ids=None, grl_lambda: float = 0.0):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        if self.injector is not None and self.inject_point == "mid":
            self._pending = feats
            try:
                out = self.encoder(**kw)
            finally:
                self._pending = None
            h = out.last_hidden_state
        else:
            out = self.encoder(**kw)
            h = out.last_hidden_state
            if self.injector is not None and feats is not None:
                h = self.injector(h, feats)
        s = masked_mean_pool(h, attention_mask)
        logits = self.classifier(s)
        result = {"logits": logits, "embedding": s}
        if self.dann:
            result["domain_logits"] = self.domain_head(grad_reverse(s, grl_lambda))
        return result


class SupConLoss(nn.Module):
    """Khosla et al. 2020, single-view formulation on L2-normalized sentence
    embeddings; positives = same sentiment label within the batch."""

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(emb.float(), dim=-1)
        sim = z @ z.t() / self.tau
        n = z.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        sim.masked_fill_(eye, -1e9)
        valid = labels >= 0
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
        pos_mask &= valid.unsqueeze(0) & valid.unsqueeze(1)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_cnt = pos_mask.sum(1)
        anchors = pos_cnt > 0
        if not anchors.any():
            return emb.new_zeros(())
        loss = -(log_prob * pos_mask).sum(1)[anchors] / pos_cnt[anchors]
        return loss.mean()


def resolve_backbone(name_or_list, cache_dir: str | None = None) -> str:
    """Return the first resolvable model id (records which candidate worked)."""
    from transformers import AutoConfig

    cands = name_or_list if isinstance(name_or_list, (list, tuple)) else [name_or_list]
    last = None
    for c in cands:
        try:
            AutoConfig.from_pretrained(c, cache_dir=cache_dir)
            return c
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"None of the backbone candidates resolved: {cands} ({last})")

"""QLoRA fine-tuning of open LLMs (Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B) on the
same gold+bulk pool, evaluated on gold-test by deterministic label scoring
(compare the summed log-probability of ' NEG' / ' NEU' / ' POS' completions).

Requires: requirements-llm.txt (peft, bitsandbytes, accelerate).
Llama-3.1 is gated on HF: set HF_TOKEN and accept the license first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .dataset import LABEL2ID
from .evaluate import full_report
from .pools import make_pools
from .thermal import ThermalGuard
from .utils import Timer, is_done, log, mark_done, set_seed, setup_logging, write_json

LABEL_TOK = {0: "NEG", 1: "NEU", 2: "POS"}
PROMPT = ("You are an expert in Azerbaijani sentiment analysis. Label the comment as NEG, NEU, or POS. "
          "Pay attention to morphological negation (e.g., -ma/-mə, -mır/-mir, deyil) and emojis.\n"
          "Comment: {text}\nLabel:")


class SFTData(Dataset):
    def __init__(self, df):
        self.text = df["text"].tolist()
        self.label = df["label"].map(LABEL2ID).tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return {"text": self.text[i][:1200], "label": self.label[i]}


class SFTCollator:
    def __init__(self, tok, max_len: int):
        self.tok = tok
        self.max_len = max_len

    def __call__(self, batch):
        prompts = [PROMPT.format(text=b["text"]) for b in batch]
        answers = [" " + LABEL_TOK[b["label"]] for b in batch]
        full = [p + a for p, a in zip(prompts, answers)]
        enc = self.tok(full, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        # mask out the prompt part: everything except the answer tokens
        for i, p in enumerate(prompts):
            p_len = len(self.tok(p, truncation=True, max_length=self.max_len)["input_ids"])
            labels[i, : min(p_len, labels.shape[1])] = -100
        enc["labels"] = labels
        return enc


def build_model(model_id: str, cache_dir: str, cfg):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, cache_dir=cache_dir,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model)
    lc = LoraConfig(r=cfg.llm.qlora.r, lora_alpha=cfg.llm.qlora.alpha, lora_dropout=cfg.llm.qlora.dropout,
                    target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def score_labels(model, tok, texts: list[str], device, batch_size: int = 8) -> np.ndarray:
    """Deterministic: pick the label whose completion has the highest summed logprob."""
    preds = np.zeros(len(texts), dtype=np.int64)
    label_ids = {c: tok(" " + LABEL_TOK[c], add_special_tokens=False)["input_ids"] for c in range(3)}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        scores = np.zeros((len(chunk), 3))
        for c in range(3):
            full = [PROMPT.format(text=t[:1200]) + " " + LABEL_TOK[c] for t in chunk]
            enc = tok(full, truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            logp = torch.log_softmax(out.logits.float(), dim=-1)
            for i in range(len(chunk)):
                ids = enc["input_ids"][i]
                n_ans = len(label_ids[c])
                seq_len = int(enc["attention_mask"][i].sum())
                s = 0.0
                for k in range(n_ans):
                    pos = seq_len - n_ans + k
                    s += float(logp[i, pos - 1, ids[pos]])
                scores[i, c] = s
        preds[start : start + len(chunk)] = scores.argmax(1)
    return preds


def run(model_key: str, limit_train: int | None = None, seed: int = 42,
        regime: str = "pooled", scope: str = "all") -> None:
    from transformers import AutoTokenizer

    cfg = load_config()
    tag = model_key if regime == "pooled" else f"{model_key}_{regime}_{scope}"
    out_dir = Path(cfg.paths.runs_dir) / "llm" / tag
    if is_done(out_dir):
        log.info("SKIP llm %s", model_key)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "qlora.log")
    set_seed(seed)
    t = Timer()
    model_id = cfg.llm.qlora_models[model_key]
    cache_dir = str(Path(cfg.paths.runs_dir) / "hf_cache")
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_df, dev_df, test_df, comp = make_pools(cfg, regime, scope, seed=seed)
    if limit_train:
        train_df = train_df.sample(n=min(limit_train, len(train_df)), random_state=seed)
    model = build_model(model_id, cache_dir, cfg)
    coll = SFTCollator(tok, cfg.llm.qlora.max_len)
    nw = int(cfg.train.get("num_workers", 0))
    loader = DataLoader(SFTData(train_df), batch_size=cfg.llm.qlora.batch_size, shuffle=True,
                        collate_fn=coll, num_workers=nw, persistent_workers=nw > 0, drop_last=True)
    accum = int(cfg.llm.qlora.grad_accum)
    steps_total = (len(loader) // accum) * int(cfg.llm.qlora.epochs)
    try:
        import bitsandbytes as bnb

        optim = bnb.optim.PagedAdamW8bit(model.parameters(), lr=cfg.llm.qlora.lr)
    except Exception:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.llm.qlora.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, steps_total))
    guard = ThermalGuard(cfg.thermal, out_dir / "thermal.csv")

    model.train()
    step = 0
    for epoch in range(int(cfg.llm.qlora.epochs)):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    guard.maybe_pause()
                if step % 100 == 0:
                    log.info("%s step %d/%d loss %.4f", model_key, step, steps_total, float(out.loss))
    model.save_pretrained(out_dir / "adapter")

    model.eval()
    preds = score_labels(model, tok, test_df["text"].tolist(), device, batch_size=4)
    y = test_df["label"].map(LABEL2ID).to_numpy()
    report = full_report(y, preds, test_df["domain"].tolist())
    pd.DataFrame({"uid": test_df["uid"], "domain": test_df["domain"], "gold": y, "pred": preds}).to_csv(
        out_dir / "preds_test.csv", index=False)
    write_json(out_dir / "metrics.json", {
        "run_id": f"llm.{tag}", "system": f"qlora_{model_key}", "model_id": model_id,
        "regime": regime, "scope": scope, "seed": seed,
        "qlora": dict(cfg.llm.qlora), "composition": comp, "test": report,
        "wall_seconds": round(t.elapsed(), 1), "thermal": guard.summary()})
    mark_done(out_dir, {"test_macro_f1": report["macro_f1"]})
    log.info("QLoRA %s (%s/%s) done: macro-F1 %.4f", model_key, regime, scope, report["macro_f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--regime", default="pooled", choices=["pooled", "lodo", "indomain"])
    ap.add_argument("--scope", default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.model_key, args.limit_train, seed=args.seed, regime=args.regime, scope=args.scope)


if __name__ == "__main__":
    main()

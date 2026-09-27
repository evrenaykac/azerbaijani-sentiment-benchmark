"""Datasets, collators and batch samplers.

Two data paths:
  * cached  - subword ids and token features come from a prebuilt TokenCache
              (see tokcache.py); the collator only pads and gathers.
  * on-the-fly - tokenizes per batch (used when no cache is available).

Batching is length-aware in both paths: examples of similar length share a
batch, so padding waste drops from ~128 tokens to roughly the local mean.
For the supervised-contrastive sampler this is done WITHIN length buckets, so
each batch still contains the same 3 labels x 2 domains structure and the
cross-domain positive-pair guarantee is unchanged.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .features import D_FEAT, SUBSETS

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]
DOM2ID = {d: i for i, d in enumerate(DOMAINS)}


# ---------------------------------------------------------------- datasets --
class SentDataset(Dataset):
    """On-the-fly path: rows with uid, text, label (str|None), domain."""

    def __init__(self, df, feature_extractor=None):
        self.uid = df["uid"].tolist()
        self.text = df["text"].tolist()
        lbl = df["label"] if "label" in df.columns else df["final_label"]
        self.label = [LABEL2ID.get(l, -100) if isinstance(l, str) else -100 for l in lbl]
        self.domain = [DOM2ID.get(d, 0) for d in df["domain"]]
        self.fx = feature_extractor
        self.lengths = [max(1, len(t.split())) for t in self.text]

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, i):
        item = {"uid": self.uid[i], "label": self.label[i], "domain": self.domain[i]}
        if self.fx is not None:
            ws, F = self.fx(self.text[i])
            item["words"], item["feats"] = ws, F
        else:
            item["text"] = self.text[i]
        return item


class CachedDataset(Dataset):
    """Cached path: everything is precomputed; __getitem__ is a slice lookup."""

    def __init__(self, df, cache, with_feats: bool):
        self.cache = cache
        self.with_feats = with_feats
        uids = df["uid"].tolist()
        lbl = df["label"] if "label" in df.columns else df["final_label"]
        labels = [LABEL2ID.get(l, -100) if isinstance(l, str) else -100 for l in lbl]
        doms = [DOM2ID.get(d, 0) for d in df["domain"]]
        keep = [i for i, u in enumerate(uids) if u in cache]
        if len(keep) != len(uids):
            missing = len(uids) - len(keep)
            raise KeyError(f"{missing} uids missing from the token cache - rebuild it (azsent.build_cache)")
        self.uid = uids
        self.label = labels
        self.domain = doms
        self.lengths = [cache.length(u) for u in uids]

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, i):
        ids, wid, feats = self.cache.get(self.uid[i])
        return {"uid": self.uid[i], "label": self.label[i], "domain": self.domain[i],
                "ids": ids, "wid": wid, "wfeats": feats}


# --------------------------------------------------------------- collators --
class Collator:
    """On-the-fly collator (tokenizes each batch)."""

    def __init__(self, tokenizer, max_len: int = 128, with_feats: bool = True):
        self.tok = tokenizer
        self.max_len = max_len
        self.with_feats = with_feats

    def __call__(self, batch):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        domains = torch.tensor([b["domain"] for b in batch], dtype=torch.long)
        uids = [b["uid"] for b in batch]
        if self.with_feats:
            enc = self.tok([b["words"] for b in batch], is_split_into_words=True, truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt")
            d_feat = batch[0]["feats"].shape[1]
            T = enc["input_ids"].shape[1]
            feats = torch.zeros(len(batch), T, d_feat, dtype=torch.float32)
            for i, b in enumerate(batch):
                wid = enc.word_ids(i)
                F = b["feats"]
                for t, w in enumerate(wid):
                    if w is not None and w < len(F):
                        feats[i, t] = torch.from_numpy(F[w])
            out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                   "feats": feats, "labels": labels, "domains": domains, "uids": uids}
        else:
            enc = self.tok([b["text"] for b in batch], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt")
            out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                   "labels": labels, "domains": domains, "uids": uids}
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        return out


class CachedCollator:
    """Pads cached ids and gathers word features onto subword positions."""

    def __init__(self, pad_id: int, with_feats: bool, subset: str = "all"):
        self.pad_id = pad_id
        self.with_feats = with_feats
        mask = np.zeros(D_FEAT, dtype=np.float32)
        mask[SUBSETS[subset]] = 1.0
        self.mask = mask

    def __call__(self, batch):
        B = len(batch)
        T = max(len(b["ids"]) for b in batch)
        input_ids = np.full((B, T), self.pad_id, dtype=np.int64)
        attn = np.zeros((B, T), dtype=np.int64)
        feats = np.zeros((B, T, D_FEAT), dtype=np.float32) if self.with_feats else None
        for i, b in enumerate(batch):
            ids = b["ids"]
            L = len(ids)
            input_ids[i, :L] = ids
            attn[i, :L] = 1
            if self.with_feats:
                w = b["wid"].astype(np.int32)
                wf = b["wfeats"]
                valid = (w >= 0) & (w < wf.shape[0])
                if valid.any():
                    pos = np.flatnonzero(valid)
                    feats[i, pos] = wf[w[pos]].astype(np.float32)
        out = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attn),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "domains": torch.tensor([b["domain"] for b in batch], dtype=torch.long),
            "uids": [b["uid"] for b in batch],
        }
        if self.with_feats:
            out["feats"] = torch.from_numpy(feats * self.mask)
        return out


# ---------------------------------------------------------------- samplers --
class LengthGroupedBatchSampler(Sampler):
    """Shuffle, chunk into mega-batches, sort each by length, emit batches,
    then shuffle batch order. Standard trick: same examples, far less padding."""

    def __init__(self, lengths, batch_size: int, seed: int = 0, mega: int = 64, drop_last: bool = False):
        self.lengths = np.asarray(lengths)
        self.bs = batch_size
        self.seed = seed
        self.mega = mega * batch_size
        self.drop_last = drop_last
        self.epoch = 0
        self.n = len(self.lengths)

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self):
        return self.n // self.bs if self.drop_last else (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        rng = np.random.RandomState(self.seed * 1000 + self.epoch)
        perm = rng.permutation(self.n)
        batches = []
        for s in range(0, self.n, self.mega):
            chunk = perm[s : s + self.mega]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk), self.bs):
                bat = chunk[b : b + self.bs]
                if self.drop_last and len(bat) < self.bs:
                    continue
                batches.append(bat.tolist())
        rng.shuffle(batches)
        return iter(batches)


class DomainStratifiedBatchSampler(Sampler):
    """Guarantees every batch contains same-label examples from >=2 domains
    (cross-domain positive pairs for every anchor). Modes:
      stratified : 3 labels x 2 domains x k samples  (paper default)
      indomain   : single domain per batch, labels mixed within it
      random     : plain shuffled batches (baseline)

    Selection happens inside a length bucket, so batches are length-homogeneous
    without changing the label/domain composition of a batch.
    """

    def __init__(self, labels, domains, batch_size: int = 32, mode: str = "stratified",
                 seed: int = 0, lengths=None, n_buckets: int = 8):
        self.labels = np.asarray(labels)
        self.domains = np.asarray(domains)
        self.bs = batch_size
        self.mode = mode
        self.seed = seed
        self.epoch = 0
        self.n = len(self.labels)
        self.n_batches = max(1, self.n // self.bs)

        if lengths is None:
            lengths = np.ones(self.n)
        lengths = np.asarray(lengths)
        order = np.argsort(lengths, kind="stable")
        n_buckets = max(1, min(n_buckets, self.n // max(1, self.bs * 2)))
        self.buckets = [b for b in np.array_split(order, n_buckets) if len(b)]

        # per-bucket cell index, plus a global fallback
        self.bucket_cells = []
        self.bucket_by_domain = []
        for b in self.buckets:
            cells = defaultdict(list)
            by_dom = defaultdict(list)
            for i in b:
                li, di = int(self.labels[i]), int(self.domains[i])
                if li >= 0:
                    cells[(li, di)].append(i)
                by_dom[di].append(i)
            self.bucket_cells.append(cells)
            self.bucket_by_domain.append(by_dom)
        self.global_cells = defaultdict(list)
        for i, (l, d) in enumerate(zip(self.labels, self.domains)):
            if l >= 0:
                self.global_cells[(int(l), int(d))].append(i)
        self.bucket_w = np.array([len(b) for b in self.buckets], dtype=float)
        self.bucket_w /= self.bucket_w.sum()

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed * 1000 + self.epoch)
        if self.mode == "random":
            # ablation baseline: plain shuffled batches, no length grouping
            yield from self._plain(rng)
            return
        if self.mode == "indomain":
            for _ in range(self.n_batches):
                bi = rng.choice(len(self.buckets), p=self.bucket_w)
                by_dom = self.bucket_by_domain[bi]
                doms = [d for d in by_dom if len(by_dom[d]) >= self.bs]
                if not doms:
                    yield rng.choice(self.n, size=self.bs, replace=False).tolist()
                    continue
                d = doms[rng.randint(len(doms))]
                yield rng.choice(by_dom[d], size=self.bs, replace=False).tolist()
            return
        k = max(1, self.bs // 6)  # 3 labels x 2 domains x k
        for _ in range(self.n_batches):
            bi = rng.choice(len(self.buckets), p=self.bucket_w)
            cells = self.bucket_cells[bi]
            batch = []
            for l in (0, 1, 2):
                doms = [d for (ll, d) in cells if ll == l and len(cells[(ll, d)]) >= 1]
                src = cells
                if len(doms) < 2:  # rare: bucket lacks the label in 2 domains
                    doms = [d for (ll, d) in self.global_cells if ll == l]
                    src = self.global_cells
                if not doms:
                    continue
                pick = rng.choice(doms, size=min(2, len(doms)), replace=False)
                for d in pick:
                    pool = src[(l, int(d))]
                    take = min(k, len(pool))
                    batch.extend(rng.choice(pool, size=take, replace=False).tolist())
            while len(batch) < self.bs:
                pool = self.buckets[bi]
                batch.append(int(pool[rng.randint(len(pool))]))
            yield batch[: self.bs]

    def _plain(self, rng):
        perm = rng.permutation(self.n)
        for b in range(self.n_batches):
            yield perm[b * self.bs : (b + 1) * self.bs].tolist()

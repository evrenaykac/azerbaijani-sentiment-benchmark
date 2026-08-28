"""Rebuild the experiment dataset export with matched label priors.

Fixes two defects of SentiAzNet_279714_LeakageFree_50_20_30.xlsx:
  1. Ten domain spellings for five domains (gold_test used short forms).
  2. gold_test label prior (28/55/17) diverges sharply from gold_train (37/19/44),
     which depresses every macro-F1 and collapses the neutral class.

Design
  * groups   : video_id_hash when present, otherwise a per-row singleton.
              No group is ever split across partitions.
  * gold     : the 10,000 labelled gold rows are re-partitioned 5000/2000/3000
              so that all three share the gold pool's own label distribution.
  * bulk     : only groups that contain a gold_dev/gold_test row are reserved;
              everything else is available for training. (The previous export
              reserved 30% of the corpus, withholding ~81k rows for no
              methodological gain.)
  * text     : already globally unique in the source; verified, not recomputed.

Run:  python3 tools/repack_dataset.py --in <xlsx> --out <xlsx>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from azsent.normalize import dedup_key, normalize_text  # noqa: E402

DOMAIN_CANON = {
    "Technology & Digital Services": "Tech",
    "Tech/Digital": "Tech",
    "Tech": "Tech",
    "Finance & Business": "Finance",
    "Finance": "Finance",
    "Social Life & Entertainment": "Social",
    "Social": "Social",
    "Retail & Lifestyle": "Retail",
    "Retail": "Retail",
    "Public Services": "Public",
    "Public": "Public",
}
LABELS = ["negative", "neutral", "positive"]
SHARES = {"gold_train": 0.50, "gold_dev": 0.20, "gold_test": 0.30}
TARGET: dict[str, int] = {}          # filled from the actual pool size after dedup


def _stable(x: str) -> int:
    return int(hashlib.blake2b(str(x).encode("utf-8"), digest_size=8).hexdigest(), 16)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse rows that share the pipeline's own deduplication key.

    The upstream export deduplicated with its own normalizer, which is not the
    one the pipeline verifies against: 2,047 keys still collided here and 801 of
    them straddled splits, because our key also strips @mentions, so
    "@a necə?" and "@b necə?" are the same comment as far as a model is
    concerned. A guarantee has to be expressed in the terms it is checked in.

    Representative priority: gold row > has source id > has a label > longer text,
    so deduplication costs the gold pool as little as possible.
    """
    df = df.copy()
    df["key"] = df["text"].astype(str).map(normalize_text).map(dedup_key)
    prio = (df["gold_role"].notna().astype(int) * 8
            + df["video_id_hash"].notna().astype(int) * 4
            + df["label"].notna().astype(int) * 2)
    order = df.assign(_p=prio, _l=df["text"].astype(str).str.len()) \
              .sort_values(["key", "_p", "_l"], ascending=[True, False, False])
    kept = order.drop_duplicates("key", keep="first").index
    removed = len(df) - len(kept)
    return df.loc[sorted(kept)].reset_index(drop=True), removed


def canon_domains(df: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(df["domain"].dropna()) - set(DOMAIN_CANON))
    if unknown:
        raise SystemExit("Bilinmeyen alan adlari: %s" % unknown)
    df = df.copy()
    df["domain"] = df["domain"].map(DOMAIN_CANON)
    return df


def assign_gold(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """New gold_role with matched (domain x label) composition in all partitions.

    Both axes matter: the label prior drives macro-F1 and the neutral class, and
    the domain mix drives every per-domain and leave-one-domain-out number. A
    quota is therefore held per (partition, domain, label) cell.
    """
    gold = df[df["gold_role"].notna()]
    share = SHARES
    cells = gold.groupby(["domain", "label"]).size()

    quota: dict[tuple[str, str, str], int] = {}
    for (dom, lab), n in cells.items():
        raw = {p: n * share[p] for p in TARGET}
        base = {p: int(np.floor(raw[p])) for p in TARGET}
        rem = int(n) - sum(base.values())
        for p in sorted(TARGET, key=lambda p: (-(raw[p] - base[p]), p))[:rem]:
            base[p] += 1
        for p in TARGET:
            quota[(p, dom, lab)] = base[p]

    grp = df["video_id_hash"].where(df["video_id_hash"].notna(), "solo_" + df["uid"])
    gold_grp = grp[gold.index]
    members: dict[str, list[int]] = defaultdict(list)
    for idx, gid in gold_grp.items():
        members[gid].append(idx)

    out = pd.Series(index=df.index, dtype=object)
    filled: dict[tuple[str, str, str], int] = defaultdict(int)
    count: dict[str, int] = defaultdict(int)

    def overshoot(p: str, cells_needed) -> float:
        return sum(max(0, n - (quota[(p, dom, lab)] - filled[(p, dom, lab)]))
                   for (dom, lab), n in cells_needed.items())

    # indivisible groups first, largest first
    multi = sorted((g for g in members if not str(g).startswith("solo_")),
                   key=lambda g: (-len(members[g]), _stable(g)))
    for gid in multi:
        idxs = members[gid]
        need = gold.loc[idxs].groupby(["domain", "label"]).size()
        p = min(TARGET, key=lambda p: (overshoot(p, need), count[p] - TARGET[p], _stable(p + gid)))
        for i in idxs:
            out[i] = p
        for k, n in need.items():
            filled[(p,) + k] += n
        count[p] += len(idxs)

    # singletons fill each cell exactly
    rng = np.random.RandomState(seed)
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for gid in members:
        if str(gid).startswith("solo_"):
            i = members[gid][0]
            by_cell[(gold.loc[i, "domain"], gold.loc[i, "label"])].append(i)
    for cell in sorted(by_cell):
        dom, lab = cell
        idxs = sorted(by_cell[cell], key=lambda i: _stable(df.loc[i, "uid"]))
        rng.shuffle(idxs)
        need = {p: max(0, quota[(p, dom, lab)] - filled[(p, dom, lab)]) for p in TARGET}
        pos = 0
        for p in sorted(TARGET, key=lambda p: (-need[p], p)):
            for i in idxs[pos:pos + need[p]]:
                out[i] = p
                filled[(p, dom, lab)] += 1
                count[p] += 1
            pos += need[p]
        for i in idxs[pos:]:                 # rounding leftovers -> most under-filled partition
            p = min(TARGET, key=lambda p: (count[p] - TARGET[p], p))
            out[i] = p
            filled[(p, dom, lab)] += 1
            count[p] += 1
    return out


def assign_splits(df: pd.DataFrame, gold_role: pd.Series, dev_frac: float = 0.05,
                  seed: int = 42) -> pd.Series:
    """train/dev/test at group level. Reserve only groups holding gold eval rows."""
    grp = df["video_id_hash"].where(df["video_id_hash"].notna(), "solo_" + df["uid"])
    split = pd.Series(index=df.index, dtype=object)
    forced: dict[str, str] = {}
    for role, part in (("gold_test", "test"), ("gold_dev", "dev"), ("gold_train", "train")):
        for gid in grp[gold_role == role].unique():
            forced.setdefault(gid, part)          # test wins, then dev, then train
    rng = np.random.RandomState(seed)
    free = sorted(set(grp) - set(forced), key=_stable)
    rng.shuffle(free)
    sizes = grp.value_counts()
    budget = dev_frac * sizes[free].sum()
    acc = 0.0
    for gid in free:
        if acc < budget:
            forced[gid] = "dev"
            acc += sizes[gid]
        else:
            forced[gid] = "train"
    return grp.map(forced)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = [("total_rows", len(df))]
    rows.append(("dedup_key_rule", "pipeline dedup_key (NFKC, mention/url strip, punctuation collapse)"))
    grp = df["video_id_hash"].where(df["video_id_hash"].notna(), "solo_" + df["uid"])
    for part in ("train", "dev", "test"):
        rows.append(("split_%s_rows" % part, int((df["split"] == part).sum()))) 
    for role in ("gold_train", "gold_dev", "gold_test"):
        g = df[df["gold_role"] == role]
        v = g["label"].value_counts(normalize=True)
        rows.append((role + "_rows", len(g)))
        for lab in LABELS:
            rows.append(("%s_%s_pct" % (role, lab[:3]), round(100 * float(v.get(lab, 0)), 2)))
    rows.append(("labelled_rows", int(df["label"].notna().sum())))
    rows.append(("video_id_coverage_pct", round(100 * float(df["video_id_hash"].notna().mean()), 2)))
    rows.append(("distinct_groups", int(grp.nunique())))
    rows.append(("rows_in_multirow_groups_pct",
                 round(100 * float(grp.map(grp.value_counts()).gt(1).mean()), 2)))
    # leakage verification
    cross_grp = grp.groupby(grp).apply(lambda s: df.loc[s.index, "split"].nunique()).max()
    rows.append(("max_splits_per_group", int(cross_grp)))
    rows.append(("text_unique", bool(df["text"].is_unique)))
    rows.append(("leakage_rule_text", "normalized text never crosses splits"))
    rows.append(("leakage_rule_group", "no source group spans two splits"))
    rows.append(("gold_prior_rule", "all three gold partitions share the gold pool label prior"))
    rows.append(("reservation_rule", "only groups containing gold eval rows are held out"))
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_excel(a.src, sheet_name="data") if a.src.endswith((".xlsx", ".xlsm")) else pd.read_parquet(a.src)
    df = canon_domains(df).reset_index(drop=True)
    n_before = len(df)
    df, n_removed = deduplicate(df)
    print("tekillestirme: %d satir dusuruldu, %d kaldi" % (n_removed, len(df)))

    pool = int(df["gold_role"].notna().sum())
    raw = {p: pool * SHARES[p] for p in SHARES}
    base = {p: int(np.floor(raw[p])) for p in SHARES}
    for p in sorted(SHARES, key=lambda p: (-(raw[p] - base[p]), p))[:pool - sum(base.values())]:
        base[p] += 1
    TARGET.clear()
    TARGET.update(base)
    print("gold havuzu %d -> hedefler %s" % (pool, TARGET))

    df["gold_role"] = assign_gold(df, seed=a.seed)
    df["split"] = assign_splits(df, df["gold_role"], dev_frac=a.dev_frac, seed=a.seed)

    # hard invariants
    grp = df["video_id_hash"].where(df["video_id_hash"].notna(), "solo_" + df["uid"])
    assert grp.groupby(grp).apply(lambda s: df.loc[s.index, "split"].nunique()).max() == 1, "grup bolmeler arasi bolundu"
    for role, part in (("gold_train", "train"), ("gold_dev", "dev"), ("gold_test", "test")):
        sub = df[df["gold_role"] == role]
        assert len(sub) == TARGET[role], "%s sayisi %d, beklenen %d" % (role, len(sub), TARGET[role])
        assert (sub["split"] == part).all(), "%s tamamen %s bolmesinde degil" % (role, part)

    aud = audit(df)
    with pd.ExcelWriter(a.dst, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="data", index=False)
        aud.to_excel(w, sheet_name="audit", index=False)
    print(aud.to_string(index=False))


if __name__ == "__main__":
    main()

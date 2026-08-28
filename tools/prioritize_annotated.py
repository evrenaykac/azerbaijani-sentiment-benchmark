"""Move the triple-annotated items into the evaluation partitions.

The annotation effort and the evaluation split were decided independently, so
the 10,000 triple-annotated items landed mostly in gold_train and in the bulk
layer: only 4.5% of gold_test had three independent annotations plus
adjudication. A benchmark whose agreement statistics describe a pool that
barely overlaps its own test set is not a benchmark a reviewer will accept.

This re-assigns gold_role WITHIN the existing gold pool so that triple-annotated
items fill gold_test first, then gold_dev, then gold_train. Two things are held
fixed while it does so:

  * every (partition x domain x label) cell keeps exactly the count it has now,
    so the matched label prior and domain mix survive untouched;
  * whole groups move together, and groups that mix gold with bulk rows are
    pinned where they are, so no video ends up straddling a split.

It also attaches the per-annotator columns and an n_annotators flag, so the
paper can report coverage per partition rather than one global number.

  python3 tools/prioritize_annotated.py --corpus <corpus.parquet> \
      --annotations <annotations.xlsx> --out <out.parquet> [--shortfall <x.xlsx>]
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
from azsent.normalize import dedup_key  # noqa: E402

ROLES = ["gold_test", "gold_dev", "gold_train"]      # priority order for annotated rows
ROLE_SPLIT = {"gold_test": "test", "gold_dev": "dev", "gold_train": "train"}
ANN_COLS = ["ann1", "ann2", "ann3", "ann4"]


def _stable(x) -> int:
    return int(hashlib.blake2b(str(x).encode("utf-8"), digest_size=8).hexdigest(), 16)


def _read(p: Path, sheet: str | None = None) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    xl = pd.ExcelFile(p)
    return xl.parse(sheet or xl.sheet_names[0])


def groups_of(df: pd.DataFrame) -> pd.Series:
    v = df["video_id_hash"].astype("string")
    return v.where(v.notna() & (v.str.len() > 0), "solo_" + df["uid"].astype(str))


def reassign(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    gold = df[df["gold_role"].notna()]
    quota = gold.groupby(["gold_role", "domain", "label"]).size().to_dict()
    quota = {(r, d, l): int(n) for (r, d, l), n in quota.items()}

    grp = df["_grp"]
    members: dict[str, list] = defaultdict(list)
    for idx, gid in grp[gold.index].items():
        members[gid].append(idx)
    # a group that also holds bulk rows cannot change split without dragging
    # unlabeled rows into the evaluation set
    pinned = {gid for gid in members if len(members[gid]) < int((grp == gid).sum())}

    out = pd.Series(index=df.index, dtype=object)
    filled: dict[tuple, int] = defaultdict(int)

    for gid in pinned:
        for i in members[gid]:
            r = df.at[i, "gold_role"]
            out[i] = r
            filled[(r, df.at[i, "domain"], df.at[i, "label"])] += 1

    def free_cap(r, need) -> int:
        """Rows of `need` that do NOT fit in r's remaining quota. 0 = perfect fit."""
        return sum(max(0, n - (quota.get((r, d, l), 0) - filled[(r, d, l)]))
                   for (d, l), n in need.items())

    free = [g for g in members if g not in pinned]
    gdf = df.loc[[i for g in free for i in members[g]]].copy()
    gdf["_g"] = df["_grp"]
    need_all = gdf.groupby(["_g", "domain", "label"]).size()
    need_of = {g: sub.droplevel(0) for g, sub in need_all.groupby(level=0)}
    ann_of = gdf.groupby("_g")["_ann3"].mean().to_dict()
    rng = np.random.RandomState(seed)

    def place(gid, r):
        for i in members[gid]:
            out[i] = r
        for (d, l), n in need_of[gid].items():
            filled[(r, d, l)] += int(n)

    # Fill the evaluation partitions first and only from annotated groups: the
    # aim is coverage where it is reported, so gold_test gets first refusal on
    # every annotated group that fits, then gold_dev, and gold_train takes what
    # is left. Best-fit-first (largest group that fits exactly) beats
    # largest-first, because one oversized group can block a whole cell.
    placed: set = set()
    for r in ROLES:
        pool = [g for g in free if g not in placed and ann_of[g] > 0.5]
        # Quota only ever shrinks, so a group that does not fit now never will:
        # one pass is enough and a rescan loop would be quadratic for nothing.
        pool.sort(key=lambda g: (-len(members[g]), _stable(g)))
        for gid in pool:
            if free_cap(r, need_of[gid]) == 0:
                place(gid, r)
                placed.add(gid)
        if r == "gold_train":
            break

    # everything still unplaced, annotated or not, fills the remaining quota
    rest = [g for g in free if g not in placed]
    rest.sort(key=lambda g: (ann_of[g], -len(members[g]), _stable(g)))
    for gid in rest:
        r = min(reversed(ROLES), key=lambda r: (free_cap(r, need_of[gid]),
                                                ROLES[::-1].index(r)))
        place(gid, r)
        placed.add(gid)

    # any cell still short (rounding) is topped up by moving nothing - quotas are
    # exact by construction, so this only reports a violation if one exists
    short = {k: quota[k] - filled[k] for k in quota if filled[k] != quota[k]}
    if short:
        print("UYARI: kota sapmasi (rounding):", dict(list(short.items())[:6]))
    return out


def verify(df: pd.DataFrame) -> dict:
    k = df["text"].astype(str).map(dedup_key)
    g = groups_of(df)
    ev, tr = df["split"].isin(["dev", "test"]), df["split"] == "train"
    return {
        "duplicate_normalized_texts": int(len(k) - k.nunique()),
        "duplicate_uid": int(len(df) - df["uid"].nunique()),
        "groups_spanning_splits": int((df.assign(_g=g).groupby("_g")["split"].nunique() > 1).sum()),
        "eval_text_in_train": len(set(k[ev]) & set(k[tr])),
        "eval_group_in_train": len(set(g[ev]) & set(g[tr])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shortfall", default=None, help="gold_test'te uclu etiketi eksik hucreler")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = _read(Path(a.corpus)).reset_index(drop=True)
    ann = _read(Path(a.annotations))
    df = df.drop(columns=[c for c in ANN_COLS + ["annotator_ids"] if c in df.columns])

    amap = ann.set_index("uid")
    df["_ann3"] = df["uid"].isin(amap.index)
    df["_grp"] = groups_of(df)
    print("korpus %d satir | uclu etiketli eslesen %d / %d"
          % (len(df), int(df["_ann3"].sum()), len(ann)))

    before = df[df["gold_role"].notna()].groupby("gold_role")["_ann3"].agg(["sum", "size"])
    df["gold_role"] = reassign(df, a.seed)
    df["split"] = [ROLE_SPLIT[r] if isinstance(r, str) else s
                   for r, s in zip(df["gold_role"], df["split"])]
    after = df[df["gold_role"].notna()].groupby("gold_role")["_ann3"].agg(["sum", "size"])

    print("\n=== uclu etiketleme kapsamasi ===")
    print(f"{'bolme':11s} {'once':>16s} {'sonra':>16s}")
    for r in ROLES:
        b, af = before.loc[r], after.loc[r]
        print(f"{r:11s} {int(b['sum']):5d}/{int(b['size']):5d} ({b['sum']/b['size']*100:5.1f}%)"
              f" {int(af['sum']):5d}/{int(af['size']):5d} ({af['sum']/af['size']*100:5.1f}%)")

    for c in ANN_COLS:
        if c in ann.columns:
            df[c] = df["uid"].map(amap[c])
    df["n_annotators"] = np.where(df["_ann3"], 3, np.where(df["label"].notna(), 1, 0))

    lab = pd.crosstab(df["gold_role"], df["label"], normalize="index") * 100
    dom = pd.crosstab(df["gold_role"], df["domain"], normalize="index") * 100
    print("\ngold etiket dagilimi (%):"); print(lab.round(2).to_string())
    print("\ngold alan dagilimi (%):"); print(dom.round(2).to_string())
    dl, dd = float((lab.max() - lab.min()).max()), float((dom.max() - dom.min()).max())
    print("\nen buyuk fark: etiket %.2f puan | alan %.2f puan" % (dl, dd))

    inv = verify(df)
    print("dogrulama:", inv)
    bad = {k: v for k, v in inv.items() if v}
    if bad or dl > 1.5 or dd > 2.0:
        raise SystemExit(f"BASARISIZ: {bad or 'bilesim toleransi asildi'}")

    if a.shortfall:
        t = df[df["gold_role"] == "gold_test"]
        miss = t[~t["_ann3"]][["uid", "domain", "label", "text", "video_id_hash", "source"]].copy()
        miss = miss.sort_values(["domain", "label"])
        for c in ("ann1", "ann2", "ann3"):
            miss[c] = ""
        with pd.ExcelWriter(a.shortfall, engine="openpyxl") as w:
            miss.to_excel(w, sheet_name="etiketlenecek", index=False)
            (miss.groupby(["domain", "label"]).size().rename("adet").reset_index()
             .to_excel(w, sheet_name="ozet", index=False))
        print("\neksik liste: %s (%d yorum)" % (a.shortfall, len(miss)))

    out = df.drop(columns=["_ann3", "_grp"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False) if a.out.endswith(".parquet") else out.to_excel(a.out, index=False)
    print("yazildi: %s (%d satir, %d sutun)" % (a.out, len(out), len(out.columns)))


if __name__ == "__main__":
    main()

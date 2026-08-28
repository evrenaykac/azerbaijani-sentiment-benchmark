"""Ingest a prepared, leakage-checked dataset export.

The previous pipeline built splits itself (data_prep -> gold -> splits). That
put deduplication, gold assembly and grouped splitting on the critical path of
every run and made several silent failure modes possible. The dataset is now
delivered already split, with the leakage rules enforced and audited at export
time (see tools/repack_dataset.py), so this step only:

  * validates the export against the contract, refusing to continue on any
    violation instead of degrading quietly;
  * re-verifies the leakage invariants independently of whatever produced the
    file, because a guarantee nobody re-checks is not a guarantee;
  * materializes the two frames the rest of the pipeline consumes.

Expected columns
  uid, domain, label, text, video_id_hash, split, gold_role, source, source_row

Outputs (under <runs>/prepared)
  gold.parquet   one row per gold instance   (split = gold_train|gold_dev|gold_test)
  bulk.parquet   training pool               (bulk_split = bulk_train|bulk_dev)
  ingest_report.json, gold_summary.json, split_counts.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, prep_dir
from .normalize import dedup_key, normalize_text, words
from .utils import log, setup_logging, write_json

DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]
LABELS = ["negative", "neutral", "positive"]
REQUIRED = ["uid", "domain", "label", "text", "video_id_hash", "split", "gold_role"]
GOLD_ROLES = {"gold_train": "train", "gold_dev": "dev", "gold_test": "test"}


def _fail(msg: str) -> None:
    raise SystemExit("VERI SOZLESMESI IHLALI: " + msg)


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        _fail("eksik sutun(lar): %s" % missing)
    unknown_dom = sorted(set(df["domain"].dropna()) - set(DOMAINS))
    if unknown_dom:
        _fail("kanonik olmayan alan adlari: %s (beklenen %s)" % (unknown_dom, DOMAINS))
    unknown_lab = sorted(set(df["label"].dropna()) - set(LABELS))
    if unknown_lab:
        _fail("bilinmeyen etiketler: %s" % unknown_lab)
    unknown_split = sorted(set(df["split"].dropna()) - {"train", "dev", "test"})
    if unknown_split:
        _fail("bilinmeyen split degerleri: %s" % unknown_split)
    unknown_role = sorted(set(df["gold_role"].dropna()) - set(GOLD_ROLES))
    if unknown_role:
        _fail("bilinmeyen gold_role degerleri: %s" % unknown_role)
    if not df["uid"].is_unique:
        _fail("uid benzersiz degil")
    if df["gold_role"].notna().any() and df.loc[df["gold_role"].notna(), "label"].isna().any():
        _fail("etiketsiz gold satiri var")
    for role, part in GOLD_ROLES.items():
        sub = df[df["gold_role"] == role]
        if len(sub) and not (sub["split"] == part).all():
            _fail("%s satirlarinin hepsi '%s' bolmesinde degil" % (role, part))


def verify_leakage(df: pd.DataFrame, key: pd.Series, group: pd.Series) -> dict:
    """Re-derive the invariants from the data; never trust the exporter's word."""
    res = {}
    dup = int(key.duplicated().sum())
    res["duplicate_normalized_texts"] = dup
    if dup:
        _fail("normalize metin %d kez tekrar ediyor; export tekillestirilmemis" % dup)
    spans = df.assign(_g=group).groupby("_g")["split"].nunique()
    res["groups_spanning_splits"] = int((spans > 1).sum())
    if res["groups_spanning_splits"]:
        _fail("%d grup birden fazla bolmeye dagilmis" % res["groups_spanning_splits"])
    ev = key[df["gold_role"].isin(["gold_dev", "gold_test"])]
    tr = key[df["split"] == "train"]
    res["eval_text_in_train"] = len(set(ev) & set(tr))
    evg = set(group[df["gold_role"].isin(["gold_dev", "gold_test"])])
    res["eval_group_in_train"] = len(evg & set(group[df["split"] == "train"]))
    if res["eval_text_in_train"] or res["eval_group_in_train"]:
        _fail("degerlendirme metni/grubu egitim havuzunda gorunuyor")
    return res


def label_table(df: pd.DataFrame, by: str) -> dict:
    out = {}
    for k, g in df.groupby(by):
        v = g["label"].value_counts()
        out[str(k)] = {"n": int(len(g)), **{l: int(v.get(l, 0)) for l in LABELS}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="hazir .xlsx/.parquet; yoksa config'ten")
    ap.add_argument("--sheet", default="data")
    args = ap.parse_args()

    cfg = load_config()
    out = prep_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "ingest.log")

    src = Path(args.source) if args.source else Path(cfg.paths.data_dir) / cfg.paths.dataset_name
    if not src.exists():
        _fail("veri dosyasi bulunamadi: %s" % src)
    log.info("Kaynak: %s", src)
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_excel(src, sheet_name=args.sheet)
    validate(df)

    df = df.reset_index(drop=True)
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["key"] = df["text"].map(dedup_key)
    df["n_words"] = df["text"].map(lambda t: len(words(t)))
    df["group_id"] = df["video_id_hash"].where(
        df["video_id_hash"].notna(), "solo_" + df["uid"].astype(str)).astype(str)
    df["has_human_label"] = df["label"].notna()

    leak = verify_leakage(df, df["key"], df["group_id"])
    log.info("Sizinti dogrulamasi gecti: %s", leak)

    gold = df[df["gold_role"].notna()].copy()
    gold["split"] = gold["gold_role"]
    gold["final_label"] = gold["label"]
    gold = gold[["uid", "text", "key", "n_words", "label", "final_label", "domain",
                 "group_id", "video_id_hash", "split", "has_human_label"]]
    gold.to_parquet(out / "gold.parquet", index=False)

    # The held-out split contributes nothing to training: gold_test lives there
    # and the few companion rows share its source groups.
    bulk = df[df["gold_role"].isna() & (df["split"] != "test")].copy()
    bulk["bulk_split"] = bulk["split"].map({"train": "bulk_train", "dev": "bulk_dev"})
    bulk["final_label"] = bulk["label"]
    bulk["video_id"] = bulk["video_id_hash"]
    bulk = bulk[["uid", "text", "key", "n_words", "label", "final_label", "domain", "group_id",
                 "video_id", "bulk_split", "has_human_label"]]
    bulk.to_parquet(out / "bulk.parquet", index=False)

    dropped = int((df["gold_role"].isna() & (df["split"] == "test")).sum())
    report = {
        "source_file": str(src), "rows_total": int(len(df)),
        "gold_rows": int(len(gold)), "bulk_rows": int(len(bulk)),
        "reserved_rows_dropped_from_training": dropped,
        "leakage_verification": leak,
        "video_id_coverage_pct": round(100 * float(df["video_id_hash"].notna().mean()), 2),
        "distinct_groups": int(df["group_id"].nunique()),
        "rows_in_multirow_groups_pct": round(
            100 * float(df["group_id"].map(df["group_id"].value_counts()).gt(1).mean()), 2),
        "bulk_human_labeled_rows": int(bulk["has_human_label"].sum()),
        "bulk_unlabeled_rows": int((~bulk["has_human_label"]).sum()),
    }
    write_json(out / "ingest_report.json", report)
    write_json(out / "gold_summary.json", {
        "counts": {r: int((gold["split"] == r).sum()) for r in GOLD_ROLES},
        "by_partition": label_table(gold, "split"),
        "by_domain": label_table(gold, "domain"),
        "by_partition_domain": {
            r: label_table(gold[gold["split"] == r], "domain") for r in GOLD_ROLES},
        "agreement": {"available": False,
                      "note": "annotator-level records were not supplied with this export"},
    })
    write_json(out / "split_counts.json", {
        "gold": {r: int((gold["split"] == r).sum()) for r in GOLD_ROLES},
        "bulk": bulk["bulk_split"].value_counts().to_dict(),
        "bulk_by_domain": bulk.groupby(["domain", "bulk_split"]).size().unstack(fill_value=0).to_dict(),
    })
    log.info("gold %d | bulk %d (insan etiketli %d) | rezerve edilip atilan %d",
             len(gold), len(bulk), report["bulk_human_labeled_rows"], dropped)


if __name__ == "__main__":
    main()

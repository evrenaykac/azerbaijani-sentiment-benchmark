"""Cross-lingual transfer + external-dataset generalization block.

Experiments (each guarded - a failing dataset download logs and continues):
  T1  Turkish-only XLM-R  -> AZ gold-test (zero-shot transfer)
  T2  LaBSE frozen + linear (TR-only, TR + AZ gold-train)
  T3  DAPT -> TR intermediate task -> AZ full framework (paper's best row)
  T4  KazSAnDRA (Kazakh): xlmr_ft vs recipe (morph cues + SupCon) in-language
  T5  Uzbek app reviews (binary): xlmr_ft vs recipe in-language
  T6  External Azerbaijani dataset: zero-shot eval of the pooled full model
      + in-dataset xlmr_ft reference

Turkish resources (named + sized in transfer_report.json for the paper):
  maydogan/Turkish_SentimentAnalysis_TRSAv1  (3-class)
  winvoker/turkish-sentiment-analysis-dataset (3-class: Positive/Negative/Notr)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config, prep_dir, runs_root
from .dataset import Collator, SentDataset, LABEL2ID, DOMAINS
from .evaluate import full_report, macro_f1
from .features import FeatureExtractor
from .normalize import normalize_text
from .thermal import ThermalGuard
from .utils import Timer, is_done, log, mark_done, set_seed, setup_logging, write_json

_TEXT_CANDS = ["text", "sentence", "review", "comment", "yorum", "Metin", "metin",
               "content", "normalized_review_text", "review_text", "body"]
_LABEL_CANDS = ["label", "sentiment", "Durum", "durum", "score", "rating", "polarity", "stars", "label_text"]

_TR_LABEL_MAP = {
    "positive": "positive", "pozitif": "positive", "olumlu": "positive", "1": "positive", "2": "positive",
    "negative": "negative", "negatif": "negative", "olumsuz": "negative", "0": "negative",
    "notr": "neutral", "nötr": "neutral", "neutral": "neutral", "tarafsız": "neutral",
}


def _norm_label_generic(v, scheme: str) -> str | None:
    s = str(v).strip().lower()
    if scheme == "tr":
        return _TR_LABEL_MAP.get(s)
    if scheme == "stars5":  # 1-2 neg, 3 neu, 4-5 pos (numeric or worded scales)
        words = {"very poor": 1.0, "poor": 2.0, "fair": 3.0, "good": 4.0, "excellent": 5.0}
        if s in words:
            x = words[s]
        else:
            try:
                x = float(s)
            except ValueError:
                return None
        return "negative" if x <= 2 else ("neutral" if x == 3 else "positive")
    if scheme == "binary":
        if s in ("1", "positive", "pos", "ijobiy"):
            return "positive"
        if s in ("0", "-1", "negative", "neg", "salbiy"):
            return "negative"
        return None
    return _TR_LABEL_MAP.get(s)


def _hf_to_df(ds_id: str, scheme: str, split_names=("train", "test", "validation")) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(ds_id)
    frames = []
    for sp in ds:
        d = ds[sp].to_pandas()
        low = {c.lower(): c for c in d.columns}
        tcol = next((low[c.lower()] for c in _TEXT_CANDS if c.lower() in low), None)
        lcol = next((low[c.lower()] for c in _LABEL_CANDS if c.lower() in low), None)
        if tcol is None or lcol is None:
            raise ValueError(f"{ds_id}/{sp}: columns not recognized: {list(d.columns)}")
        part = pd.DataFrame({"text_raw": d[tcol].astype(str), "label": d[lcol].map(lambda v: _norm_label_generic(v, scheme))})
        part["hf_split"] = sp
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].notna()].copy()
    df["text"] = df["text_raw"].map(normalize_text)
    df = df[df["text"].str.len() >= 3].drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["uid"] = [f"{ds_id.split('/')[-1]}_{i}" for i in range(len(df))]
    df["domain"] = "Tech"  # placeholder domain id for the collator; not used in metrics
    return df


def _split_frame(df: pd.DataFrame, seed: int = 42, frac=(0.8, 0.1, 0.1)) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n1 = int(len(df) * frac[0])
    n2 = int(len(df) * (frac[0] + frac[1]))
    return {"train": df.iloc[idx[:n1]], "dev": df.iloc[idx[n1:n2]], "test": df.iloc[idx[n2:]]}


def _train_encoder(cfg, train_df, dev_df, backbone_id, out_dir: Path, fx=None, supcon=False,
                   epochs=None, seed=42, save_encoder=False, init_dir=None):
    """Compact trainer reused by transfer experiments (CE +optional SupCon)."""
    from transformers import AutoTokenizer

    from .modeling import SentModel, SupConLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    set_seed(seed)
    cache_dir = str(Path(cfg.paths.runs_dir) / "hf_cache")
    tok = AutoTokenizer.from_pretrained(init_dir or backbone_id, cache_dir=cache_dir, use_fast=True)
    with_feats = fx is not None
    coll = Collator(tok, cfg.train.max_len, with_feats=with_feats)
    model = SentModel(backbone_id, inject=("film" if with_feats else "none"), cache_dir=cache_dir,
                      local_dir=init_dir).to(device)
    tr = DataLoader(SentDataset(train_df, fx), batch_size=cfg.train.batch_size, shuffle=True, collate_fn=coll,
                    num_workers=0, drop_last=False)
    dv = DataLoader(SentDataset(dev_df, fx), batch_size=cfg.train.eval_batch_size, collate_fn=coll, num_workers=0)
    n_ep = epochs or cfg.train.epochs
    total = max(1, len(tr)) * n_ep
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=0.01)
    warm = int(total * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    ce = torch.nn.CrossEntropyLoss()
    sc = SupConLoss(cfg.train.supcon_tau)
    guard = ThermalGuard(cfg.thermal, out_dir / "thermal.csv")
    best, best_state, bad = -1, None, 0
    for ep in range(n_ep):
        model.train()
        for i, b in enumerate(tr):
            b.pop("uids")
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(b["input_ids"], b["attention_mask"], feats=b.get("feats"),
                            token_type_ids=b.get("token_type_ids"))
                loss = ce(out["logits"].float(), b["labels"])
                if supcon:
                    loss = loss + cfg.train.supcon_lambda * sc(out["embedding"], b["labels"])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            if i % 50 == 0:
                guard.maybe_pause()
        # dev eval
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in dv:
                b.pop("uids")
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(b["input_ids"], b["attention_mask"], feats=b.get("feats"),
                                token_type_ids=b.get("token_type_ids"))
                ys.append(b["labels"].cpu().numpy())
                ps.append(out["logits"].float().argmax(1).cpu().numpy())
        f1 = macro_f1(np.concatenate(ys), np.concatenate(ps))
        log.info("  epoch %d dev F1 %.4f", ep, f1)
        if f1 > best + 1e-5:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.train.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    if save_encoder:
        (out_dir / "encoder").mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(out_dir / "encoder")
        tok.save_pretrained(out_dir / "encoder")
    return model, tok, best


@torch.no_grad()
def _eval_model(cfg, model, tok, test_df, fx=None):
    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    coll = Collator(tok, cfg.train.max_len, with_feats=fx is not None)
    loader = DataLoader(SentDataset(test_df, fx), batch_size=cfg.train.eval_batch_size, collate_fn=coll, num_workers=0)
    ys, ps, ds = [], [], []
    model.eval()
    for b in loader:
        b.pop("uids")
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(b["input_ids"], b["attention_mask"], feats=b.get("feats"),
                        token_type_ids=b.get("token_type_ids"))
        ys.append(b["labels"].cpu().numpy())
        ps.append(out["logits"].float().argmax(1).cpu().numpy())
        ds.append(b["domains"].cpu().numpy())
    y, p, d = np.concatenate(ys), np.concatenate(ps), np.concatenate(ds)
    return full_report(y, p, [DOMAINS[i] for i in d])


def _az_gold_test(cfg):
    gold = pd.read_parquet(prep_dir(cfg) / "gold.parquet")
    te = gold[gold["split"] == "gold_test"].copy()
    te["label"] = te["final_label"]
    return te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of t1..t6")
    args = ap.parse_args()
    cfg = load_config()
    out_root = Path(cfg.paths.runs_dir) / "transfer"
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "transfer.log")
    only = set(args.only.split(",")) if args.only else None
    results = {}
    az_test = _az_gold_test(cfg)
    cache = str(Path(cfg.paths.runs_dir) / "hf_cache")

    # ---- Turkish pool -------------------------------------------------------
    tr_pool = None
    try:
        tr1 = _hf_to_df(cfg.transfer.turkish.trsav1, "tr")
        tr2 = _hf_to_df(cfg.transfer.turkish.winvoker, "tr")
        tr_pool = pd.concat([tr1.assign(src="TRSAv1"), tr2.assign(src="winvoker")], ignore_index=True)
        tr_pool = tr_pool.drop_duplicates(subset=["text"]).reset_index(drop=True)
        results["turkish_datasets"] = {
            "TRSAv1": {"id": cfg.transfer.turkish.trsav1, "n": int(len(tr1)),
                       "labels": tr1["label"].value_counts().to_dict()},
            "winvoker": {"id": cfg.transfer.turkish.winvoker, "n": int(len(tr2)),
                         "labels": tr2["label"].value_counts().to_dict()},
        }
    except Exception as e:  # noqa: BLE001
        log.warning("Turkish datasets unavailable: %s", e)

    if tr_pool is not None and (only is None or {"t1", "t3"} & only):
        sp = _split_frame(tr_pool)
        if only is None or "t1" in only:
            d = out_root / "t1_tr_only"
            if not is_done(d):
                d.mkdir(exist_ok=True)
                model, tok, _ = _train_encoder(cfg, sp["train"], sp["dev"], "xlm-roberta-base", d, epochs=3)
                rep = _eval_model(cfg, model, tok, az_test)
                results["t1_tr_only_to_az"] = rep
                write_json(d / "metrics.json", rep)
                mark_done(d)
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if only is None or "t3" in only:
            d = out_root / "t3_tr_intermediate"
            if not is_done(d):
                d.mkdir(exist_ok=True)
                dapt_dir = Path(cfg.paths.runs_dir) / "dapt" / "xlmr_all" / "model"
                init = str(dapt_dir) if dapt_dir.exists() else None
                _, _, _ = _train_encoder(cfg, sp["train"], sp["dev"], "xlm-roberta-base", d,
                                         epochs=2, save_encoder=True, init_dir=init)
                # stage 2: full framework on AZ with the TR-adapted encoder
                import subprocess, sys

                r = subprocess.run([sys.executable, "-m", "azsent.train_clf", "--system", "full",
                                    "--regime", "pooled", "--scope", "all", "--seed", "42",
                                    "--tag", "trstage", "--no-dapt",
                                    "--backbone-local", str(d / "encoder")], check=False)
                results["t3_tr_intermediate"] = {"stage2_returncode": r.returncode}
                mark_done(d)

    # ---- T2: LaBSE frozen + linear -----------------------------------------
    if tr_pool is not None and (only is None or "t2" in only):
        d = out_root / "t2_labse"
        if not is_done(d):
            d.mkdir(exist_ok=True)
            try:
                from transformers import AutoModel, AutoTokenizer
                from sklearn.linear_model import LogisticRegression

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tok = AutoTokenizer.from_pretrained(cfg.backbones.labse, cache_dir=cache)
                enc = AutoModel.from_pretrained(cfg.backbones.labse, cache_dir=cache).float().to(device).eval()

                @torch.no_grad()
                def embed(texts):
                    out = []
                    for s in range(0, len(texts), 128):
                        e = tok(texts[s:s + 128], truncation=True, max_length=128, padding=True,
                                return_tensors="pt").to(device)
                        o = enc(**e)
                        out.append(torch.nn.functional.normalize(o.pooler_output, dim=-1).cpu().numpy())
                    return np.concatenate(out)

                sp = _split_frame(tr_pool)
                tr_sub = sp["train"].sample(n=min(60000, len(sp["train"])), random_state=42)
                Xtr, ytr = embed(tr_sub["text"].tolist()), tr_sub["label"].map(LABEL2ID).to_numpy()
                Xte, yte = embed(az_test["text"].tolist()), az_test["label"].map(LABEL2ID).to_numpy()
                clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, ytr)
                rep_tr_only = full_report(yte, clf.predict(Xte), az_test["domain"].tolist())
                gold = pd.read_parquet(prep_dir(cfg) / "gold.parquet")
                gtr = gold[gold["split"] == "gold_train"]
                Xaz, yaz = embed(gtr["text"].tolist()), gtr["final_label"].map(LABEL2ID).to_numpy()
                clf2 = LogisticRegression(max_iter=2000, random_state=42).fit(
                    np.concatenate([Xtr, Xaz]), np.concatenate([ytr, yaz]))
                rep_mixed = full_report(yte, clf2.predict(Xte), az_test["domain"].tolist())
                results["t2_labse"] = {"tr_only": rep_tr_only, "tr_plus_az_goldtrain": rep_mixed}
                write_json(d / "metrics.json", results["t2_labse"])
                mark_done(d)
                del enc
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:  # noqa: BLE001
                log.warning("T2 LaBSE failed: %s", e)

    # ---- T4/T5: in-language Turkic recipe tests -----------------------------
    for tag, ds_id, scheme, lang in (("t4_kazakh", cfg.transfer.kazakh, "stars5", "kk"),
                                     ("t5_uzbek", cfg.transfer.get("uzbek_hf", "risqaliyevds/uzbek-sentiment-analysis"), "stars5", "uz")):
        if only is not None and tag.split("_")[0] not in only:
            continue
        d = out_root / tag
        if is_done(d):
            continue
        try:
            df = _hf_to_df(ds_id, scheme)
            if len(df) < 500:
                raise ValueError(f"too few usable rows ({len(df)})")
            df = df.sample(n=min(len(df), 60000), random_state=42).reset_index(drop=True)
            sp = _split_frame(df)
            d.mkdir(exist_ok=True)
            m1, tok1, _ = _train_encoder(cfg, sp["train"], sp["dev"], "xlm-roberta-base", d / "base", epochs=3)
            rep_base = _eval_model(cfg, m1, tok1, sp["test"])
            del m1
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            fx = FeatureExtractor({}, lang=lang, subset="morph")
            m2, tok2, _ = _train_encoder(cfg, sp["train"], sp["dev"], "xlm-roberta-base", d / "recipe",
                                         fx=fx, supcon=True, epochs=3)
            rep_recipe = _eval_model(cfg, m2, tok2, sp["test"], fx=fx)
            del m2
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            results[tag] = {"dataset": ds_id, "n_used": int(len(df)),
                            "labels": df["label"].value_counts().to_dict(),
                            "xlmr_ft": rep_base, "recipe_morph_supcon": rep_recipe}
            write_json(d / "metrics.json", results[tag])
            mark_done(d)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", tag, e)

    # ---- T6: external Azerbaijani dataset -----------------------------------
    if only is None or "t6" in only:
        d = out_root / "t6_az_external"
        if not is_done(d):
            try:
                df = _hf_to_df(cfg.transfer.azerbaijani_external, "stars5")
                sp = _split_frame(df)
                d.mkdir(exist_ok=True)
                m1, tok1, _ = _train_encoder(cfg, sp["train"], sp["dev"], "xlm-roberta-base", d / "base", epochs=3)
                rep_ft = _eval_model(cfg, m1, tok1, sp["test"])
                del m1
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                # Decisive external check: the full recipe (polarity + morphology + SupCon) on an
                # independent Azerbaijani dataset, where the SentiAzNet lexicon does apply.
                from .lexicon import load_lexicon

                fx6 = FeatureExtractor(load_lexicon(cfg.paths.data_dir, cfg.paths.lexicon_subdir),
                                       lang="az", subset="all",
                                       match=str(cfg.get("lexicon", {}).get("match", "prefix")),
                                       min_prefix_len=int(cfg.get("lexicon", {}).get("min_prefix_len", 4)))
                m2, tok2, _ = _train_encoder(cfg, sp["train"], sp["dev"], "xlm-roberta-base",
                                             d / "recipe", fx=fx6, supcon=True, epochs=3)
                rep_recipe = _eval_model(cfg, m2, tok2, sp["test"], fx=fx6)
                del m2
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                results["t6_az_external"] = {"dataset": cfg.transfer.azerbaijani_external,
                                             "n_used": int(len(df)), "xlmr_ft_in_dataset": rep_ft,
                                             "recipe_pol_morph_supcon": rep_recipe}
                write_json(d / "metrics.json", results["t6_az_external"])
                mark_done(d)
            except Exception as e:  # noqa: BLE001
                log.warning("t6 failed: %s", e)

    prev = {}
    rp = out_root / "transfer_report.json"
    if rp.exists():
        try:
            prev = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    prev.update(results)          # --only ile kismi kosuda eski sonuclari EZME
    write_json(rp, prev)
    log.info("Transfer block finished; report: %s", out_root / "transfer_report.json")


if __name__ == "__main__":
    main()

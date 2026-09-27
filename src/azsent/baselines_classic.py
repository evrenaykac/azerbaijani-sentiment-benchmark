"""Classic baselines: TF-IDF (word+char) + linear classifier, and
fastText-style embeddings (gensim) + MLP. Same pools/splits/seeds as the
neural systems; outputs the same preds_test.csv/metrics.json contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, runs_root
from .dataset import LABEL2ID, DOMAINS, DOM2ID
from .evaluate import full_report
from .pools import make_pools
from .utils import Timer, is_done, log, mark_done, set_seed, setup_logging, write_json


def _dump(run_dir: Path, test_df, y, pred, probs, meta):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "uid": test_df["uid"], "domain": test_df["domain"], "gold": y, "pred": pred,
        "logit_neg": np.log(np.clip(probs[:, 0], 1e-9, 1)), "logit_neu": np.log(np.clip(probs[:, 1], 1e-9, 1)),
        "logit_pos": np.log(np.clip(probs[:, 2], 1e-9, 1)),
        "p_neg": probs[:, 0], "p_neu": probs[:, 1], "p_pos": probs[:, 2],
    }).to_csv(run_dir / "preds_test.csv", index=False)
    report = full_report(y, pred, test_df["domain"].tolist())
    meta["test"] = report
    write_json(run_dir / "metrics.json", meta)
    mark_done(run_dir, {"test_macro_f1": report["macro_f1"]})
    return report


def run_tfidf(args) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    cfg = load_config()
    run_id = f"{args.regime}.{args.scope}.tfidf_linear.s{args.seed}"
    run_dir = runs_root(cfg) / run_id
    if is_done(run_dir) and not args.force:
        log.info("SKIP %s", run_id)
        return
    set_seed(args.seed)
    t = Timer()
    train_df, dev_df, test_df, comp = make_pools(cfg, args.regime, args.scope, seed=args.seed)
    if args.limit_train:
        train_df = train_df.sample(n=min(args.limit_train, len(train_df)), random_state=args.seed)
    feats = FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, random_state=args.seed)
    pipe = Pipeline([("f", feats), ("clf", clf)])
    y_tr = train_df["label"].map(LABEL2ID).to_numpy()
    pipe.fit(train_df["text"], y_tr)
    y_te = test_df["label"].map(LABEL2ID).to_numpy()
    probs = pipe.predict_proba(test_df["text"])
    pred = probs.argmax(1)
    rep = _dump(run_dir, test_df, y_te, pred, probs, {
        "run_id": run_id, "system": "tfidf_linear", "regime": args.regime, "scope": args.scope,
        "seed": args.seed, "composition": comp,
        "hyperparams": {"word_ngrams": "1-2", "char_wb_ngrams": "2-5", "min_df": 2,
                        "max_features": "300k+300k", "clf": "LogisticRegression(C=4.0, max_iter=2000)"},
        "wall_seconds": round(t.elapsed(), 1)})
    log.info("DONE %s macro-F1 %.4f", run_id, rep["macro_f1"])


def run_fasttext(args) -> None:
    cfg = load_config()
    run_id = f"{args.regime}.{args.scope}.fasttext_mlp.s{args.seed}"
    run_dir = runs_root(cfg) / run_id
    if is_done(run_dir) and not args.force:
        log.info("SKIP %s", run_id)
        return
    try:
        from gensim.models import FastText
    except Exception as e:  # noqa: BLE001
        log.error("gensim unavailable (%s). Install requirements-classic or skip fastText baseline.", e)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "metrics.json", {"run_id": run_id, "error": f"gensim unavailable: {e}"})
        return
    from sklearn.neural_network import MLPClassifier

    set_seed(args.seed)
    t = Timer()
    train_df, dev_df, test_df, comp = make_pools(cfg, args.regime, args.scope, seed=args.seed)
    if args.limit_train:
        train_df = train_df.sample(n=min(args.limit_train, len(train_df)), random_state=args.seed)
    sents = [t.split() for t in train_df["text"].tolist()]
    ft = FastText(vector_size=200, window=5, min_count=2, workers=4, seed=args.seed, epochs=8)
    ft.build_vocab(sents)
    ft.train(sents, total_examples=len(sents), epochs=8)

    def emb(texts):
        out = np.zeros((len(texts), 200), dtype=np.float32)
        for i, t_ in enumerate(texts):
            vs = []
            for w in t_.split():
                try:
                    vs.append(ft.wv[w])  # fastText resolves OOV words via subword n-grams
                except KeyError:
                    continue
            if vs:
                out[i] = np.mean(vs, axis=0)
        return out

    Xtr, Xte = emb(train_df["text"].tolist()), emb(test_df["text"].tolist())
    y_tr = train_df["label"].map(LABEL2ID).to_numpy()
    y_te = test_df["label"].map(LABEL2ID).to_numpy()
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, random_state=args.seed, early_stopping=True)
    mlp.fit(Xtr, y_tr)
    probs = mlp.predict_proba(Xte)
    rep = _dump(run_dir, test_df, y_te, probs.argmax(1), probs, {
        "run_id": run_id, "system": "fasttext_mlp", "regime": args.regime, "scope": args.scope,
        "seed": args.seed, "composition": comp,
        "hyperparams": {"fasttext": "gensim FastText(dim=200, win=5, min_count=2, epochs=8, subwords default)",
                        "mlp": "MLP(256), early_stopping, max_iter=60"},
        "wall_seconds": round(t.elapsed(), 1)})
    log.info("DONE %s macro-F1 %.4f", run_id, rep["macro_f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["tfidf_linear", "fasttext_mlp"])
    ap.add_argument("--regime", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging()
    if args.system == "tfidf_linear":
        run_tfidf(args)
    else:
        run_fasttext(args)


if __name__ == "__main__":
    main()

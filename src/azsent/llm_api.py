"""Prompted proprietary LLM evaluation (GPT-4o / GPT-4o-mini, pinned
snapshots) on gold-test, zero-shot and 3-shot, deterministic decoding.
Requires OPENAI_API_KEY in the environment; every request/response is logged.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, prep_dir
from .dataset import LABEL2ID
from .evaluate import full_report
from .utils import Timer, is_done, log, mark_done, setup_logging, write_json

SYS_PROMPT = ("You are an expert in Azerbaijani sentiment analysis. Label each comment as NEG, NEU, or POS.\n"
              "Pay attention to morphological negation (e.g., -ma/-mə, -mır/-mir, deyil) and emojis/punctuation.\n"
              "Return only labels in the format ID: LABEL.")
LAB2ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID2LAB = {0: "NEG", 1: "NEU", 2: "POS"}
_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.\-]\s*(NEG|NEU|POS)\b", re.I | re.M)


def _fewshot_block(gold: pd.DataFrame, seed: int = 42) -> str:
    rng = np.random.RandomState(seed)
    lines = []
    for lab, tag in (("negative", "NEG"), ("neutral", "NEU"), ("positive", "POS")):
        cand = gold[(gold["split"] == "gold_train") & (gold["final_label"] == lab)]
        row = cand.iloc[rng.randint(len(cand))]
        lines.append(f"Example ({tag}): {row['text'][:300]}")
    return "\n".join(lines)


def eval_model(client, model_id: str, test: pd.DataFrame, shots: int, batch: int, out_jsonl: Path,
               fewshot: str | None) -> np.ndarray:
    import json

    preds = np.full(len(test), -1, dtype=np.int64)
    texts = test["text"].tolist()
    with open(out_jsonl, "a", encoding="utf-8") as jf:
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            body = "\n".join(f"{i+1}: {t[:800]}" for i, t in enumerate(chunk))
            user = (fewshot + "\n\n" if shots and fewshot else "") + body
            for attempt in range(2):
                try:
                    r = client.chat.completions.create(
                        model=model_id, temperature=0,
                        messages=[{"role": "system", "content": SYS_PROMPT},
                                  {"role": "user", "content": user}])
                    txt = r.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001
                    log.warning("API error (%s), retrying in 10s", e)
                    time.sleep(10)
                    continue
                jf.write(json.dumps({"model": model_id, "start": start, "response": txt}, ensure_ascii=False) + "\n")
                got = {}
                for m in _LINE_RE.finditer(txt):
                    got[int(m.group(1))] = LAB2ID[m.group(2).upper()]
                if len(got) >= len(chunk) * 0.8 or attempt == 1:
                    for i in range(len(chunk)):
                        preds[start + i] = got.get(i + 1, -1)
                    break
    # unparsed -> counted as errors: assign an always-wrong pseudo-label
    bad = preds == -1
    if bad.any():
        gold_ids = test["final_label"].map(LABEL2ID).to_numpy()
        preds[bad] = (gold_ids[bad] + 1) % 3
        log.warning("%d/%d outputs invalid after retry (counted as errors)", int(bad.sum()), len(preds))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg.paths.runs_dir) / "llm" / "openai"
    if is_done(out_dir):
        log.info("SKIP openai eval")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set - skipping the prompted-LLM evaluation (set the key and rerun).")
        return
    from openai import OpenAI

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "openai.log")
    t = Timer()
    client = OpenAI()
    gold = pd.read_parquet(prep_dir(cfg) / "gold.parquet")
    test = gold[gold["split"] == "gold_test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[: args.limit]
    fewshot = _fewshot_block(gold)
    results = {}
    y = test["final_label"].map(LABEL2ID).to_numpy()
    for key, model_id in cfg.llm.openai_models.items():
        for shots in (0, 3):
            tag = f"{key}_{shots}shot"
            preds = eval_model(client, model_id, test, shots, int(cfg.llm.openai_batch),
                               out_dir / f"{tag}.jsonl", fewshot if shots else None)
            rep = full_report(y, preds, test["domain"].tolist())
            results[tag] = {"model_id": model_id, "shots": shots, "test": rep}
            pd.DataFrame({"uid": test["uid"], "domain": test["domain"], "gold": y, "pred": preds}).to_csv(
                out_dir / f"preds_{tag}.csv", index=False)
            log.info("%s: macro-F1 %.4f", tag, rep["macro_f1"])
    write_json(out_dir / "metrics.json", {"run_id": "llm.openai", "results": results,
                                          "eval_date": time.strftime("%Y-%m-%d"),
                                          "wall_seconds": round(t.elapsed(), 1)})
    mark_done(out_dir)


if __name__ == "__main__":
    main()

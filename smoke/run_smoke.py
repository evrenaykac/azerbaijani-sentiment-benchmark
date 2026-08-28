"""End-to-end smoke test for the v2 pipeline.

Exercises the real code path on a tiny sampled workspace:

    ingest -> truncation -> teacher/silver -> DAPT -> train_clf
    (xlmr_ft, full, full_midinject, gold_only, silver ablation, strict LODO)
    -> classic baseline -> postproc -> stats -> report

plus smoke/selftest_arch.py, which proves each ablation switch actually changes
the computation (a no-op switch produces a table that looks fine and is wrong).

    python smoke/run_smoke.py --source <hazir_veri.parquet> --workdir <dir>
    python smoke/run_smoke.py ... --real-backbone   # xlm-roberta-base (GPU)

Without --real-backbone a randomly-initialised 4-layer model is built locally,
so nothing is downloaded and the whole thing runs on CPU in a few minutes. The
scores are meaningless by design - this checks that every stage runs, writes
what the next stage reads, and produces the final tables.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATASET = "smoke_dataset.xlsx"


def run(cmd: list[str], env: dict) -> None:
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode != 0:
        print(f"\nSMOKE FAILURE in: {' '.join(map(str, cmd))}")
        sys.exit(1)


def build_tiny_model(dest: Path, texts: list[str]) -> None:
    """A 4-layer randomly-initialised XLM-R clone with a BPE tokenizer trained on
    the smoke texts themselves.

    Nothing is downloaded: the smoke test has to work on a machine with no HF
    access (a locked-down lab box, an offline pod), otherwise the one check that
    is supposed to run *before* the expensive run is the first thing to fail.
    """
    if (dest / "config.json").exists() and (dest / "tokenizer.json").exists():
        print(f"tiny model zaten var: {dest}")
        return
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForMaskedLM

    dest.mkdir(parents=True, exist_ok=True)
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=4000, special_tokens=specials))
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", tk.token_to_id("<s>")), ("</s>", tk.token_to_id("</s>"))])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>", cls_token="<s>", sep_token="</s>",
        model_max_length=512)
    tok.save_pretrained(dest)
    cfg = XLMRobertaConfig(vocab_size=tok.vocab_size, hidden_size=128,
                           num_hidden_layers=4, num_attention_heads=4,
                           intermediate_size=256, max_position_embeddings=514,
                           bos_token_id=tk.token_to_id("<s>"),
                           eos_token_id=tk.token_to_id("</s>"),
                           pad_token_id=tk.token_to_id("<pad>"))
    XLMRobertaForMaskedLM(cfg).save_pretrained(dest)
    print(f"tiny model yazildi: {dest} (vocab {tok.vocab_size}, indirme yok)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="hazir veri seti .xlsx/.parquet")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--lexicon", default=None, help="sozluk dosyasi (varsayilan: <veri>/lexicon/*)")
    ap.add_argument("--real-backbone", action="store_true")
    args = ap.parse_args()

    wd = Path(args.workdir).resolve()
    data_dir, runs_dir = wd / "data", wd / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    env = dict(os.environ)
    env["AZSENT_DATA"] = str(data_dir)
    env["AZSENT_RUNS"] = str(runs_dir)
    env["AZSENT_DATASET"] = DATASET      # config override; default.yaml is NOT edited
    env["PYTHONIOENCODING"] = "utf-8"

    # --- data ---------------------------------------------------------------
    run([py, HERE / "make_smoke.py", "--source", args.source, "--out", data_dir], env)

    lex_dir = data_dir / "lexicon"
    lex_dir.mkdir(parents=True, exist_ok=True)
    src_lex = [Path(args.lexicon)] if args.lexicon else \
        sorted((Path(args.source).resolve().parent / "lexicon").glob("*"))
    for f in src_lex:
        if f.is_file():
            shutil.copy2(f, lex_dir / f.name)
    if not any(lex_dir.iterdir()):
        print(f"SMOKE FAILURE: sozluk bulunamadi -> {lex_dir}  (--lexicon ile verin)")
        sys.exit(1)

    tiny: list[str] = []
    if not args.real_backbone:
        import pandas as pd

        sm = pd.read_excel(data_dir / DATASET, sheet_name="data")
        build_tiny_model(data_dir / "tiny_model", sm["text"].astype(str).tolist())
        tiny = ["--backbone-local", str(data_dir / "tiny_model")]
        # cheap architectural check before the pipeline: an ablation switch that
        # silently does nothing would still produce a full, plausible-looking
        # table, so it has to be caught here rather than in the results.
        run([py, HERE / "selftest_arch.py", "--model", data_dir / "tiny_model"], env)

    # --- pipeline -----------------------------------------------------------
    run([py, "-m", "azsent.ingest"], env)
    run([py, "-m", "azsent.lexcov"], env)
    run([py, "-m", "azsent.truncation", *tiny], env)
    run([py, "-m", "azsent.teacher", "--scope", "all", "--epochs", "1", *tiny], env)
    run([py, "-m", "azsent.teacher", "--scope", "lodo_Public", "--epochs", "1", *tiny], env)
    run([py, "-m", "azsent.train_dapt", "--scope", "all", "--steps", "30", *tiny], env)
    run([py, "-m", "azsent.train_dapt", "--scope", "lodo_Public", "--steps", "30", *tiny], env)

    common = ["--seed", "13", "--epochs", "1", "--smoke", *tiny]
    run([py, "-m", "azsent.train_clf", "--system", "xlmr_ft", "--regime", "lodo", "--scope", "Public", *common], env)
    run([py, "-m", "azsent.train_clf", "--system", "full", "--regime", "lodo", "--scope", "Public", *common], env)
    run([py, "-m", "azsent.train_clf", "--system", "full_midinject", "--regime", "lodo", "--scope", "Public", *common], env)
    run([py, "-m", "azsent.train_clf", "--system", "xlmr_ft", "--regime", "indomain", "--scope", "Finance", *common], env)
    run([py, "-m", "azsent.train_clf", "--system", "full", "--regime", "pooled", "--scope", "all",
         "--silver-frac", "0.5", "--tag", "frac050", *common], env)
    run([py, "-m", "azsent.train_clf", "--system", "full", "--regime", "lodo", "--scope", "Public",
         "--pool-mode", "gold_only", "--tag", "goldonly", *common], env)
    run([py, "-m", "azsent.train_clf", "--system", "xlmr_ft", "--regime", "lodo", "--scope", "Public",
         "--strict-lodo", "--silver-frac", "1.0", "--tag", "strict", *common], env)
    run([py, "-m", "azsent.baselines_classic", "--system", "tfidf_linear", "--regime", "indomain",
         "--scope", "Finance", "--seed", "13"], env)

    run([py, "-m", "azsent.postproc"], env)
    run([py, "-m", "azsent.run_stats", "--smoke"], env)
    run([py, "-m", "azsent.report"], env)

    tables = sorted((runs_dir / "report").glob("*"))
    if not tables:
        print(f"\nSMOKE FAILURE: rapor uretilmedi -> {runs_dir / 'report'}")
        sys.exit(1)

    print("\n================= SMOKE TESTI GECTI =================")
    print(f"Cikti klasoru : {runs_dir}")
    print(f"Rapor dosyalari: {len(tables)} adet")
    for t in tables:
        print("  -", t.name)


if __name__ == "__main__":
    main()

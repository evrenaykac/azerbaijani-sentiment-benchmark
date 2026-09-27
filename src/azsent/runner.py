"""Experiment queue runner: builds the full job graph, executes sequentially
with thermal gating, resumes from DONE markers, tracks wall time.

Usage
  python -m azsent.runner --list --blocks core            # show the plan + ETA
  python -m azsent.runner --blocks core                   # run the core block
  python -m azsent.runner --blocks core,sensitivity --max-hours 10
  python -m azsent.runner --blocks all

Blocks
  prep        data prep + gold + splits + truncation + coverage
  teachers    fold-safe teachers + silver labels (all + 5 LODO folds)
  dapt        DAPT checkpoints (all + 5 LODO folds)
  indomain    in-domain table systems (5 domains x 5 seeds)
  lodo        LODO table systems (5 targets x 3 seeds)
  ablation    extra single-component ablations (R1 item 6, mid-injection)
  sensitivity silver-frac grid, data-strategy, sampler, lambda/tau
  strict      strict LODO robustness check (teacher re-labels human layer)
  postproc    probes + calibration for every finished run
  stats       significance tests (cluster bootstrap, Holm)
  report      aggregate tables (CSV + LaTeX)
  llm         QLoRA fine-tunes (+ optional OpenAI eval if key set)
  transfer    Turkish / Turkic / external-AZ experiments
  core        prep+teachers+dapt+indomain+lodo+postproc+stats+report
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config, load_systems, prep_dir, runs_root
from .thermal import ThermalGuard
from .utils import is_done, log, setup_logging, write_json

DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]


class Job:
    def __init__(self, jid: str, module: str, args: list[str], block: str, est_min: float,
                 done_path: Path | None = None, done_file: Path | None = None):
        self.jid = jid
        self.module = module
        self.args = args
        self.block = block
        self.est_min = est_min
        self.done_path = done_path
        self.done_file = done_file

    def is_done(self) -> bool:
        if self.done_file is not None:
            return self.done_file.exists()
        return bool(self.done_path and is_done(self.done_path))

    def argv(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]


def build_jobs(cfg, systems, blocks: set[str], smoke: bool = False) -> list[Job]:
    R = runs_root(cfg)
    P = prep_dir(cfg)
    jobs: list[Job] = []
    seeds_in = cfg.train.seeds_indomain
    seeds_lodo = cfg.train.seeds_lodo
    if smoke:
        seeds_in, seeds_lodo = [13], [13]
    # Headline systems get the full seed budget; context rows use 3 seeds.
    seeds_ctx = seeds_in[:3] if len(seeds_in) > 3 else seeds_in
    headline_sys = set(systems.tables.get("headline_indomain", ["xlmr_ft", "full", "mdeberta_ft"]))
    # Same idea under LODO. v1 ran every LODO cell on 3 seeds, and the two
    # comparisons that mattered came back with 95% CIs about 13 points wide -
    # wide enough that only the largest effect cleared zero. The systems the
    # claims rest on get the full seed budget here; context rows stay at 3.
    seeds_lodo_head = cfg.train.get("seeds_lodo_headline", seeds_lodo)
    if smoke:
        seeds_lodo_head = [13]
    headline_lodo = set(systems.tables.get("headline_lodo", []))

    def train_job(system, regime, scope, seed, block, est, extra=None, tag=""):
        rid = f"{regime}.{scope}.{system}.s{seed}" + (f".{tag}" if tag else "")
        args = ["--system", system, "--regime", regime, "--scope", scope, "--seed", str(seed)]
        if tag:
            args += ["--tag", tag]
        if extra:
            args += extra
        jobs.append(Job(rid, "azsent.train_clf", args, block, est, R / rid))

    # ---- prep ---------------------------------------------------------------
    if "prep" in blocks:
        # One ingest step replaces data_prep -> gold -> splits: the export already
        # carries audited splits, so this validates and materializes rather than
        # recomputing three fragile stages on every run.
        jobs.append(Job("prep.ingest", "azsent.ingest", [], "prep", 4, done_file=P / "ingest_report.json"))
        jobs.append(Job("prep.lexicon", "azsent.lexcov", [], "prep", 1, done_file=P / "lexicon_coverage.json"))
        jobs.append(Job("prep.truncation", "azsent.truncation", [], "prep", 8, done_file=P / "truncation_report.json"))
        # tokenize + featurize the whole corpus once; every training run reuses it
        cache_backbones = ["xlmr"]
        if {"indomain", "ablation"} & blocks:
            cache_backbones += ["mdeberta", "mbert", "berturk"]
        jobs.append(Job("prep.cache", "azsent.build_cache", ["--backbones", *cache_backbones],
                        "prep", 12, done_file=Path(cfg.paths.runs_dir) / "cache"))

    # ---- teachers -----------------------------------------------------------
    if "teachers" in blocks:
        jobs.append(Job("teacher.all", "azsent.teacher", ["--scope", "all"], "teachers", 10,
                        Path(cfg.paths.runs_dir) / "teachers" / "all"))
        for t in DOMAINS:
            jobs.append(Job(f"teacher.lodo_{t}", "azsent.teacher", ["--scope", f"lodo_{t}"], "teachers", 9,
                            Path(cfg.paths.runs_dir) / "teachers" / f"lodo_{t}"))

    # ---- dapt ---------------------------------------------------------------
    if "dapt" in blocks:
        jobs.append(Job("dapt.all", "azsent.train_dapt", ["--scope", "all"], "dapt", 12,
                        Path(cfg.paths.runs_dir) / "dapt" / "xlmr_all"))
        if {"indomain", "ablation"} & blocks:  # full_mbert / full_mdeberta need these
            jobs.append(Job("dapt.mbert_all", "azsent.train_dapt", ["--scope", "all", "--backbone", "mbert"],
                            "dapt", 8, Path(cfg.paths.runs_dir) / "dapt" / "mbert_all"))
            # DeBERTa-v3 is RTD-pretrained, so this stage also trains a fresh MLM head; and
            # its checkpoint ships in fp16, which recent transformers honours by default and
            # which silently diverges - train_dapt/modeling force fp32 (see .float()).
            jobs.append(Job("dapt.mdeberta_all", "azsent.train_dapt", ["--scope", "all", "--backbone", "mdeberta"],
                            "dapt", 13, Path(cfg.paths.runs_dir) / "dapt" / "mdeberta_all"))
        for t in DOMAINS:
            jobs.append(Job(f"dapt.lodo_{t}", "azsent.train_dapt", ["--scope", f"lodo_{t}"], "dapt", 11,
                            Path(cfg.paths.runs_dir) / "dapt" / f"xlmr_lodo_{t}"))
        if "lodo" in blocks and "full_mbert" in list(systems.tables.get("lodo", [])):
            for t in DOMAINS:
                jobs.append(Job(f"dapt.mbert_lodo_{t}", "azsent.train_dapt",
                                ["--scope", f"lodo_{t}", "--backbone", "mbert"], "dapt", 8,
                                Path(cfg.paths.runs_dir) / "dapt" / f"mbert_lodo_{t}"))
        # Without these, every mDeBERTa system with dapt:true is missing its
        # fold-safe checkpoint under LODO and its rows never appear in the table -
        # which is exactly how full_mdeberta went missing from the first v4 run.
        _mdeb_lodo = {"full_mdeberta", "mdeberta_dapt"} & set(systems.tables.get("lodo", []))
        if "lodo" in blocks and _mdeb_lodo:
            for t in DOMAINS:
                jobs.append(Job(f"dapt.mdeberta_lodo_{t}", "azsent.train_dapt",
                                ["--scope", f"lodo_{t}", "--backbone", "mdeberta"], "dapt", 13,
                                Path(cfg.paths.runs_dir) / "dapt" / f"mdeberta_lodo_{t}"))

    # ---- headline fast-track: the paper's main-claim rows first --------------
    if "headline" in blocks:
        for system in ("xlmr_ft", "full"):
            for seed in seeds_in:
                for d in DOMAINS:
                    train_job(system, "indomain", d, seed, "headline", 4,
                              extra=(["--save-model"] if system == "full" and seed == seeds_in[0] else None))
        # The paper's main claim is a LODO claim, so the fast-track carries the
        # full seed budget for the arms it rests on - a 3-seed core_fast would
        # reproduce exactly the wide intervals that made v1's result fragile.
        # xlmr_supcon is in the fast-track because "does the recipe beat the
        # contrastive objective alone?" is now a headline question.
        for system in ("xlmr_ft", "full", "xlmr_supcon"):
            for seed in seeds_lodo_head:
                for t in DOMAINS:
                    train_job(system, "lodo", t, seed, "headline", 5)
        for seed in seeds_lodo:
            for t in DOMAINS:
                train_job("xlmr_dapt_film", "lodo", t, seed, "headline", 5)

    # ---- in-domain ----------------------------------------------------------
    if "indomain" in blocks:
        for seed in seeds_in:
            for d in DOMAINS:
                for cls in ("tfidf_linear", "fasttext_mlp"):
                    rid = f"indomain.{d}.{cls}.s{seed}"
                    jobs.append(Job(rid, "azsent.baselines_classic",
                                    ["--system", cls, "--regime", "indomain", "--scope", d, "--seed", str(seed)],
                                    "indomain", 2, R / rid))
        priority = ["xlmr_ft", "full", "xlmr_dapt", "xlmr_film", "xlmr_add", "mdeberta_ft",
                    "full_mdeberta", "full_mbert", "mbert_ft", "berturk_ft", "allma_ft"]
        order = [s for s in priority if s in systems.tables.indomain] + \
                [s for s in systems.tables.indomain if s not in priority]
        for system in order:
            for seed in (seeds_in if system in headline_sys else seeds_ctx):
                for d in DOMAINS:
                    est = 6 if "mdeberta" in system else 4
                    train_job(system, "indomain", d, seed, "indomain", est,
                              extra=(["--save-model"] if system == "full" and seed == seeds_in[0] else None))

    # ---- lodo ---------------------------------------------------------------
    if "lodo" in blocks:
        priority = ["xlmr_ft", "full", "xlmr_dapt_film", "xlmr_film", "xlmr_dapt", "xlmr_dann",
                    "mdeberta_ft", "allma_ft"]
        order = [s for s in priority if s in systems.tables.lodo] + \
                [s for s in systems.tables.lodo if s not in priority]
        for system in order:
            for seed in (seeds_lodo_head if system in headline_lodo else seeds_lodo):
                for t in DOMAINS:
                    est = 7 if "mdeberta" in system else 5
                    train_job(system, "lodo", t, seed, "lodo", est)

    # ---- extra ablations ----------------------------------------------------
    if "ablation" in blocks:
        for system in systems.tables.indomain_extra_ablation:
            for seed in seeds_ctx:
                for d in DOMAINS:
                    train_job(system, "indomain", d, seed, "ablation", 4)
        for system in systems.tables.lodo_extra_ablation:
            for seed in seeds_lodo:
                for t in DOMAINS:
                    train_job(system, "lodo", t, seed, "ablation", 5)

    # ---- data strategy + hyper-parameter sensitivity -------------------------
    if "sensitivity" in blocks:
        sens_seeds = seeds_lodo
        ds = dict(systems.tables.get("data_strategy", {}) or {})
        ds_regimes = list(ds.get("regimes", ["pooled"]))
        ds_modes = list(ds.get("pool_modes", ["gold+bulk"]))
        ds_fracs = list(ds.get("silver_fracs", [0.0]))

        def strategy_jobs(system, modes, fracs, est):
            """(pool_mode x silver_frac) under every requested regime.

            v1 varied this pooled only, so an 8.6-point LODO effect stayed
            invisible until the run was over. Both regimes are measured here.
            """
            for seed in sens_seeds:
                for rgm in ds_regimes:
                    scopes = DOMAINS if rgm == "lodo" else ["all"]
                    for scope in scopes:
                        for pm in modes:
                            # gold_only has no silver portion to vary
                            for frac in ([0.0] if pm == "gold_only" else fracs):
                                tag = "%s_f%03d" % (pm.replace("+", ""), int(frac * 100))
                                train_job(system, rgm, scope, seed, "sensitivity", est,
                                          extra=["--pool-mode", pm, "--silver-frac", str(frac)],
                                          tag=tag)

        strategy_jobs("full", ds_modes, ds_fracs, 6)
        # the baseline only needs the two endpoints, to show the effect is not
        # a property of the proposed system alone
        strategy_jobs("xlmr_ft", ds_modes, [0.0], 5)

        for seed in sens_seeds:
            for lam in systems.tables.sensitivity_lambda_tau["lambdas"]:
                for tau in systems.tables.sensitivity_lambda_tau["taus"]:
                    if abs(lam - 0.1) < 1e-9 and abs(tau - 0.07) < 1e-9:
                        continue          # equals the reference configuration
                    train_job("full", "pooled", "all", seed, "sensitivity", 7,
                              extra=["--supcon-lambda", str(lam), "--supcon-tau", str(tau)],
                              tag=f"lam{lam}_tau{tau}")
            for target in systems.tables.get("sampler_targets", ["Public", "Tech"]):
                for system in ("full_sampler_random", "full_sampler_indomain"):
                    train_job(system, "lodo", target, seed, "sensitivity", 5)

    # ---- strict LODO robustness check ---------------------------------------
    if "strict" in blocks:
        for system in ("full",):
            for t in DOMAINS:
                train_job(system, "lodo", t, seeds_lodo[0], "strict", 6,
                          extra=["--strict-lodo"], tag="strict")

    # ---- postproc / stats / report ------------------------------------------
    if "postproc" in blocks:
        jobs.append(Job("postproc.all", "azsent.postproc", [], "postproc", 45, None))
    if "stats" in blocks:
        jobs.append(Job("stats.all", "azsent.run_stats", [], "stats", 35, None))
    if "report" in blocks:
        jobs.append(Job("report.all", "azsent.report", [], "report", 5, None))

    # ---- llm ----------------------------------------------------------------
    if "llm" in blocks:
        for key in ("qwen25", "llama31", "qwen3"):
            out = Path(cfg.paths.runs_dir) / "llm" / key
            jobs.append(Job(f"llm.qlora.{key}", "azsent.llm_qlora", ["--model-key", key], "llm", 300, out))
        # The paper's claim is robustness under domain shift, but v1 only had
        # pooled LLM numbers, so "a 7B model beats the method" could not be
        # answered where it mattered. One model over the folds settles it
        # without paying for the full 3x5 grid.
        if cfg.llm.get("run_lodo", False):
            for key in cfg.llm.get("lodo_models", ["qwen25"]):
                for t in cfg.llm.get("lodo_targets", DOMAINS):
                    out = Path(cfg.paths.runs_dir) / "llm" / f"{key}_lodo_{t}"
                    jobs.append(Job(f"llm.qlora.{key}.lodo_{t}", "azsent.llm_qlora",
                                    ["--model-key", key, "--regime", "lodo", "--scope", t],
                                    "llm", 300, out))
        jobs.append(Job("llm.openai", "azsent.llm_api", [], "llm", 30,
                        Path(cfg.paths.runs_dir) / "llm" / "openai"))

    # ---- transfer -----------------------------------------------------------
    if "transfer" in blocks:
        T = Path(cfg.paths.runs_dir) / "transfer"
        for key, sub, est in (("t1", "t1_tr_only", 60), ("t2", "t2_labse", 45),
                              ("t3", "t3_tr_intermediate", 150), ("t4", "t4_kazakh", 120),
                              ("t5", "t5_uzbek", 60), ("t6", "t6_az_external", 60)):
            jobs.append(Job(f"transfer.{key}", "azsent.transfer", ["--only", key],
                            "transfer", est, T / sub))

    # de-duplicate by job id (headline and core tables share run ids)
    seen: set[str] = set()
    out: list[Job] = []
    for j in jobs:
        if j.jid not in seen:
            seen.add(j.jid)
            out.append(j)
    return out


BLOCK_SETS = {
    # core_fast answers the main claim in a few hours; core adds the full tables;
    # all adds ablations, the data-strategy grid, LLMs and transfer.
    "core_fast": ["prep", "teachers", "dapt", "headline", "postproc", "stats", "report"],
    "core": ["prep", "teachers", "dapt", "headline", "indomain", "lodo", "postproc", "stats", "report"],
    "all": ["prep", "teachers", "dapt", "indomain", "lodo", "ablation", "sensitivity", "strict",
            "postproc", "stats", "report", "llm", "transfer"],
}

# blocks whose jobs may be split across parallel workers with --shard
SHARDABLE = {"teachers", "dapt", "headline", "indomain", "lodo", "ablation",
             "sensitivity", "strict", "llm", "transfer"}


def resolve_blocks(spec: str) -> set[str]:
    out: list[str] = []
    for b in spec.split(","):
        b = b.strip()
        out.extend(BLOCK_SETS.get(b, [b]))
    return set(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="core")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None,
                    help="run only every N-th shardable job, offset I (parallel workers); "
                         "non-shardable jobs (prep/postproc/stats/report) run on shard 0 only")
    ap.add_argument("--gpu", type=int, default=None, help="pin this worker to CUDA device index")
    ap.add_argument("--only", default=None,
                    help="run only jobs from these blocks, while --blocks still determines "
                         "what the plan contains (so prerequisites are sized for the whole run)")
    args = ap.parse_args()
    cfg = load_config()
    shard_tag = f".s{args.shard[0]}" if args.shard else ""
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / f"runner{shard_tag}.log")
    systems = load_systems()
    blocks = resolve_blocks(args.blocks)
    jobs = build_jobs(cfg, systems, blocks, smoke=args.smoke)
    if args.only:
        keep = {b.strip() for b in args.only.split(",")}
        jobs = [j for j in jobs if j.block in keep]
        log.info("Filtered to blocks %s: %d jobs", sorted(keep), len(jobs))
    if args.shard:
        i, n = args.shard
        assert 0 <= i < n, "--shard I N requires 0 <= I < N"
        # Cost-aware assignment (longest-processing-time first): each shardable
        # job goes to the currently lightest shard. Deterministic, so a restart
        # reproduces the same assignment; round-robin left one worker ~30%
        # longer than the others because job costs differ by 100x.
        shardable = [j for j in jobs if j.block in SHARDABLE]
        order = sorted(range(len(shardable)), key=lambda k: (-shardable[k].est_min, shardable[k].jid))
        load = [0.0] * n
        owner = {}
        for k in order:
            t = min(range(n), key=lambda x: (load[x], x))
            owner[id(shardable[k])] = t
            load[t] += shardable[k].est_min
        jobs = [j for j in jobs
                if (owner.get(id(j)) == i if j.block in SHARDABLE else i == 0)]
        log.info("Shard %d/%d: %d jobs, ~%.1f GPU-hours (shard loads: %s)",
                 i, n, len(jobs), load[i] / 60, [round(x / 60, 1) for x in load])

    pend = [j for j in jobs if not j.is_done()]
    done = len(jobs) - len(pend)
    total_min = sum(j.est_min for j in pend)
    log.info("Queue: %d jobs (%d already done), pending ETA ~ %.1f GPU-hours", len(jobs), done, total_min / 60)
    if args.list or args.dry_run:
        by_block: dict[str, list[Job]] = {}
        for j in pend:
            by_block.setdefault(j.block, []).append(j)
        for b, js in by_block.items():
            print(f"\n== {b}: {len(js)} jobs, ~{sum(x.est_min for x in js)/60:.1f} h ==")
            for j in js[:8]:
                print("   ", j.jid)
            if len(js) > 8:
                print(f"    ... +{len(js)-8} more")
        return

    import os

    child_env = os.environ.copy()
    if args.gpu is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        child_env["AZSENT_NVML_INDEX"] = str(args.gpu)
        os.environ["AZSENT_NVML_INDEX"] = str(args.gpu)  # runner's own gate too
    guard = ThermalGuard(cfg.thermal, Path(cfg.paths.runs_dir) / "logs" / f"thermal_runner{shard_tag}.csv")
    state_path = Path(cfg.paths.runs_dir) / f"queue_state{shard_tag}.json"
    log_csv = Path(cfg.paths.runs_dir) / f"runs_log{shard_tag}.csv"
    t0 = time.time()
    failures = []
    for i, j in enumerate(jobs):
        if j.is_done():
            continue
        if args.max_hours and (time.time() - t0) / 3600 >= args.max_hours:
            log.info("Reached --max-hours %.1f, stopping cleanly. Re-run the same command to resume.", args.max_hours)
            break
        guard.gate_job_start()
        write_json(state_path, {"current": j.jid, "index": i, "total": len(jobs),
                                "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        log.info("[%d/%d] START %s (est %.0f min)", i + 1, len(jobs), j.jid, j.est_min)
        ok, secs = _run_once(j, child_env)
        if not ok:
            log.warning("Job %s failed once - retrying", j.jid)
            ok, secs2 = _run_once(j, child_env)
            secs += secs2
        status = "ok" if ok else "FAILED"
        if not ok:
            failures.append(j.jid)
        new = not log_csv.exists()
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["job", "block", "status", "seconds", "finished_at"])
            w.writerow([j.jid, j.block, status, round(secs, 1), time.strftime("%Y-%m-%d %H:%M:%S")])
    write_json(state_path, {"current": None, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "failures": failures})
    log.info("Queue finished. Failures: %s", failures or "none")


def _run_once(j: Job, env: dict | None = None) -> tuple[bool, float]:
    t = time.time()
    try:
        r = subprocess.run(j.argv(), check=False, env=env)
        return r.returncode == 0, time.time() - t
    except Exception as e:  # noqa: BLE001
        log.error("Job %s crashed: %s", j.jid, e)
        return False, time.time() - t


if __name__ == "__main__":
    main()

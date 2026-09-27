"""Standard significance-test batteries for the paper tables."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, load_systems
from .stats_tests import compare_systems
from .utils import log, setup_logging

DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "stats.log")
    systems = load_systems()
    seeds_in = [13] if args.smoke else cfg.train.seeds_indomain
    # Headline LODO systems run 5 seeds and context rows 3, so the battery has to
    # offer the union: compare_systems keeps only the runs that exist, which means
    # a 5-seed system is tested on 5 and a 3-seed one on 3. Passing the short list
    # would have silently thrown away two seeds of the arms the claims rest on.
    seeds_lodo = ([13] if args.smoke else
                  sorted(set(cfg.train.seeds_lodo)
                         | set(cfg.train.get("seeds_lodo_headline", cfg.train.seeds_lodo))))

    indomain_sys = [s for s in systems.tables.indomain if s != "xlmr_ft"]
    lodo_sys = [s for s in systems.tables.lodo if s != "xlmr_ft"]
    try:
        compare_systems("indomain", "xlmr_ft", indomain_sys, seeds_in, DOMAINS, out_name="stats_indomain_vs_xlmr")
    except Exception as e:  # noqa: BLE001
        log.warning("indomain stats failed: %s", e)
    try:
        compare_systems("lodo", "xlmr_ft", lodo_sys, seeds_lodo, DOMAINS, out_name="stats_lodo_vs_xlmr")
    except Exception as e:  # noqa: BLE001
        log.warning("lodo stats failed: %s", e)
    # full vs strongest alternatives (family for the moderated claims)
    for rgm, seeds in (("indomain", seeds_in), ("lodo", seeds_lodo)):
        try:
            compare_systems(rgm, "mdeberta_ft", ["full"], seeds, DOMAINS, out_name=f"stats_{rgm}_full_vs_mdeberta")
        except Exception as e:  # noqa: BLE001
            log.warning("%s full-vs-mdeberta stats failed: %s", rgm, e)

    # Does anything beyond the contrastive objective earn its place?
    # In v1 the LODO ablation put xlmr_supcon at +2.28 over xlmr_ft and the full
    # recipe at +2.47 - i.e. SupCon alone carried almost all of it, and on the
    # one domain where the method won, SupCon alone was actually ahead. That
    # comparison was never tested, so a reviewer would have been the first to
    # run it. It is a standing part of the battery now.
    supcon_arms = ["full", "full_midinject", "xlmr_film", "xlmr_dapt"]
    for rgm, seeds, table in (("indomain", seeds_in, systems.tables.indomain),
                              ("lodo", seeds_lodo, systems.tables.lodo)):
        arms = [s for s in supcon_arms if s in table]
        if not arms:
            continue
        try:
            compare_systems(rgm, "xlmr_supcon", arms, seeds, DOMAINS,
                            out_name=f"stats_{rgm}_vs_supcon")
        except Exception as e:  # noqa: BLE001
            log.warning("%s vs-supcon stats failed: %s", rgm, e)
    log.info("stats battery done")


if __name__ == "__main__":
    main()

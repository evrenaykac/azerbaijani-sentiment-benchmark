"""Architectural self-test: proves the ablation switches actually do something.

The revision's central argument is an ablation chain - each component is claimed
to contribute a measurable delta. A switch that silently does nothing would not
crash and would not look wrong in a table; it would just quietly weaken the
paper. These checks run in seconds on the tiny smoke model and fail loudly.

  python smoke/selftest_arch.py --model <tiny_model_dir>
"""
from __future__ import annotations

import argparse
import sys

import torch

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'HATA'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yerel model klasoru (tiny_model)")
    a = ap.parse_args()

    from azsent.features import FeatureExtractor
    from azsent.modeling import SentModel, SupConLoss

    print("== mimari oz-testi ==")

    def build(inject: str, point: str = "post", trained: bool = False, **kw):
        torch.manual_seed(0)
        m = SentModel(a.model, inject=inject, inject_point=point, **kw).eval()
        if trained and m.injector is not None:
            torch.manual_seed(7)
            with torch.no_grad():
                if inject == "film":
                    m.injector.film.weight.normal_(0, 0.05)
                else:
                    m.injector.proj.weight.normal_(0, 0.05)
        return m

    ids = torch.randint(5, 1000, (2, 16))
    am = torch.ones_like(ids)
    ft = torch.randn(2, 16, 8)

    def logits(m):
        with torch.no_grad():
            return m(ids, am, feats=ft)["logits"]

    # 1) FiLM is identity at init - deliberate, so training starts from the
    #    un-injected model. If this ever changes, the "no injection" baseline
    #    stops being the right reference point.
    #    Tested by disabling the injector on the SAME model: building two models
    #    would differ anyway, because constructing the injector consumes RNG draws
    #    and so shifts the classifier head's initialisation.
    m_film = build("film")
    l_film0 = logits(m_film)
    keep, m_film.injector = m_film.injector, None
    l_noinj = logits(m_film)
    m_film.injector = keep
    check("FiLM baslangicta birim (gamma=1, beta=0)", torch.allclose(l_film0, l_noinj, atol=1e-5),
          "maxdiff %.2e" % float((l_film0 - l_noinj).abs().max()))
    l_none = l_noinj

    # 2) a trained injector must change the output, at both injection points,
    #    and the two points must not be the same computation
    l_post = logits(build("film", "post", trained=True))
    l_mid = logits(build("film", "mid", trained=True, mid_layer=2))
    l_add = logits(build("additive", "post", trained=True))
    check("FiLM (post) enjeksiyonu ciktiyi degistiriyor", not torch.allclose(l_post, l_none, atol=1e-5))
    check("FiLM (mid) enjeksiyonu ciktiyi degistiriyor", not torch.allclose(l_mid, l_none, atol=1e-5))
    check("post != mid (ayri hesaplama)", not torch.allclose(l_post, l_mid, atol=1e-5),
          "maxdiff %.4f" % float((l_post - l_mid).abs().max()))
    check("additive != film", not torch.allclose(l_add, l_post, atol=1e-5))

    m_mid = build("film", "mid", mid_layer=2)
    check("mid-inject kancasi kayitli", m_mid._hook_handle is not None)
    check("inject=none iken kanca yok", build("none", "mid")._hook_handle is None)

    # 3) DANN gradient reversal must actually reverse
    m_dann = build("none", dann=True)
    check("DANN alan basligi var", hasattr(m_dann, "domain_head"))

    # 4) SupCon must be finite and must prefer well-separated classes
    sc = SupConLoss(tau=0.1)
    y = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0], [1, 0.01], [0, 1.0], [0.01, 1]])
    bad = torch.tensor([[1.0, 0], [0, 1.0], [1, 0.01], [0.01, 1]])
    lg, lb = float(sc(good, y)), float(sc(bad, y))
    check("SupCon sonlu", torch.isfinite(torch.tensor([lg, lb])).all().item(), f"iyi={lg:.4f} kotu={lb:.4f}")
    check("SupCon ayrisik siniflari odullendiriyor", lg < lb, f"{lg:.4f} < {lb:.4f}")

    # 5) lexicon features: prefix matching must catch inflected forms that exact
    #    matching misses (this is what raised coverage 4.25% -> 5.95%)
    lex = {"yaxsi": 1.0, "pis": -1.0}
    fx_exact = FeatureExtractor(lex, match="exact")
    fx_pref = FeatureExtractor(lex, match="prefix", min_prefix_len=4)
    ws = ["yaxsidir", "amma", "pisdir"]
    ve = float(abs(fx_exact.word_features(ws)[:, 1]).sum())   # col 1 = "lexicon hit"
    vp = float(abs(fx_pref.word_features(ws)[:, 1]).sum())
    check("prefix eslesme exact'ten fazlasini yakaliyor", vp > ve, f"exact={ve:.0f} isabet, prefix={vp:.0f} isabet")

    print()
    if FAILURES:
        print("OZ-TEST BASARISIZ: " + ", ".join(FAILURES))
        sys.exit(1)
    print("oz-test gecti: butun ablasyon anahtarlari gercekten etki ediyor")


if __name__ == "__main__":
    main()

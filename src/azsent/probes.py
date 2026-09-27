"""Embedding-space probes (paper Tables 7/12 + UMAP figure).

Reads embeddings.npz from a run directory and writes probes.json (+ umap.png).
Probes: NN@{1,5,10,20} (gold-test queries vs gold-dev pool, cosine), k-means
NMI/ARI, silhouette, Davies-Bouldin, anisotropy (mean pairwise cosine),
TwoNN intrinsic dimensionality, cross-domain centroid alignment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils import log, setup_logging, write_json

DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def nn_consistency(test_emb, test_y, dev_emb, dev_y, ks=(1, 5, 10, 20)) -> dict:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=max(ks), metric="cosine").fit(dev_emb)
    _, idx = nn.kneighbors(test_emb)
    out = {}
    for k in ks:
        neigh = dev_y[idx[:, :k]]
        out[f"nn@{k}"] = float((neigh == test_y[:, None]).mean())
    return out


def clustering_metrics(emb, y, seed: int = 42) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(emb)
    return {
        "nmi": float(normalized_mutual_info_score(y, km.labels_)),
        "ari": float(adjusted_rand_score(y, km.labels_)),
        "silhouette": float(silhouette_score(emb, y, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(emb, y)),
    }


def anisotropy(emb, n_sample: int = 2000, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(emb), size=min(n_sample, len(emb)), replace=False)
    z = _l2(emb[idx])
    sim = z @ z.T
    n = len(z)
    return float((sim.sum() - n) / (n * (n - 1)))


def twonn_id(emb, seed: int = 42, discard: float = 0.1) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3).fit(emb)
    d, _ = nn.kneighbors(emb)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    mu = mu[mu > 1.0]
    n = len(mu)
    if n < 10:
        return float("nan")
    keep = int(n * (1 - discard))
    mu = mu[:keep]
    F = (np.arange(1, keep + 1) - 0.5) / n
    x = np.log(mu)
    yv = -np.log(1 - F)
    d_hat = float((x @ yv) / (x @ x))
    return d_hat


def centroid_alignment(emb, y, dom) -> dict:
    z = _l2(emb)
    cents = {}
    for d in np.unique(dom):
        for c in np.unique(y):
            m = (dom == d) & (y == c)
            if m.sum() >= 3:
                cents[(int(d), int(c))] = _l2(z[m].mean(0, keepdims=True))[0]
    within, cross = [], []
    keys = list(cents)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (d1, c1), (d2, c2) = keys[i], keys[j]
            s = float(cents[keys[i]] @ cents[keys[j]])
            if d1 != d2 and c1 == c2:
                within.append(s)
            elif d1 != d2 and c1 != c2:
                cross.append(s)
    return {"centroid_within_class_cross_domain": float(np.mean(within)) if within else None,
            "centroid_cross_class": float(np.mean(cross)) if cross else None}


def umap_plot(emb, y, out_png: Path, seed: int = 42) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap

        red = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
        xy = red.fit_transform(emb)
        colors = {0: "#c0392b", 1: "#7f8c8d", 2: "#27ae60"}
        names = {0: "negative", 1: "neutral", 2: "positive"}
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for c in (0, 1, 2):
            m = y == c
            ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.5, c=colors[c], label=names[c], linewidths=0)
        ax.legend(markerscale=4)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("UMAP plot failed: %s", e)


def run_probes(run_dir: str | Path, with_umap: bool = False) -> dict:
    run_dir = Path(run_dir)
    z = np.load(run_dir / "embeddings.npz", allow_pickle=True)
    te, ty, td = z["test_emb"].astype(np.float32), z["test_y"], z["test_d"]
    de, dy = z["dev_emb"].astype(np.float32), z["dev_y"]
    out = {}
    out.update(nn_consistency(te, ty, de, dy))
    out.update(clustering_metrics(te, ty))
    out["anisotropy"] = anisotropy(te)
    out["twonn_id"] = twonn_id(te)
    out.update(centroid_alignment(te, ty, td))
    write_json(run_dir / "probes.json", out)
    if with_umap:
        umap_plot(te, ty, run_dir / "umap.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--umap", action="store_true")
    args = ap.parse_args()
    setup_logging()
    r = run_probes(args.run_dir, args.umap)
    log.info("probes: %s", r)


if __name__ == "__main__":
    main()

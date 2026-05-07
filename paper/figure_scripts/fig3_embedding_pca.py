"""Figure 3 — 2-D PCA of BeMAE embeddings coloured by profile class.

Embedding space is morphology-aware (section Results)?  Project z(128D) 
into 2 principal components and colour each point by the 5-class profile label. 

Input
-----
* ``data/embeddings_halpha_seed42/star_data.pt``: dict[star] -> {mjds, embeddings, snrs, ews}
* ``downstream/results/spectral_features.pt``: dict[star] -> {..., profile_type_id}

Output
------
- ``figures/fig3_embedding_pca.pdf``
- ``figures/fig3_embedding_pca.png``

"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, PROFILE_COL, save_fig, setup_style  # noqa: E402

import os
EMB_FILE  = Path(os.environ.get("BESS_EMB_PATH",
                                str(ROOT / "data" / "embeddings_halpha_seed42" / "star_data.pt")))
FEAT_FILE = Path(os.environ.get("BESS_FEATURES_PATH",
                                str(ROOT / "reference_results" / "features" / "spectral_features.pt")))

PROFILE_NAMES = ["absorption_pure", "single_peak", "shell",
                 "double_peak", "double_asym"]
MAX_POINTS = 8000   # for readability of scatter


def main() -> None:
    setup_style()
    emb_data  = torch.load(EMB_FILE,  map_location="cpu", weights_only=False)
    feat_data = torch.load(FEAT_FILE, map_location="cpu", weights_only=False)

    # ---- gather all embeddings with matching profile labels ---------------
    Zs, labels = [], []
    for star, d in emb_data.items():
        if star not in feat_data:
            continue
        z = d["embeddings"].numpy()
        # Features are aligned with the same ordering of MJDs (same pipeline).
        pid = feat_data[star]["profile_type_id"].numpy()
        n = min(len(z), len(pid))
        Zs.append(z[:n])
        labels.append(pid[:n])
    Z = np.concatenate(Zs, axis=0)
    y = np.concatenate(labels, axis=0)
    print(f"[info] {Z.shape[0]} spectra with labels")

    # ---- 2-D PCA on the FULL embedding set (matches paper text) ---------
    Zc_full = Z - Z.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2, random_state=0).fit(Zc_full)
    var_ratio = pca.explained_variance_ratio_
    print(f"[info] PC1={100*var_ratio[0]:.1f}%  PC2={100*var_ratio[1]:.1f}%")

    # ---- subsample for readable scatter ---------------------------------
    rng = np.random.default_rng(42)
    keep = []
    for c in range(5):
        idx = np.where(y == c)[0]
        if len(idx) > MAX_POINTS // 5:
            idx = rng.choice(idx, MAX_POINTS // 5, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    proj = pca.transform(Zc_full[keep])
    y_s = y[keep]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for c, name in enumerate(PROFILE_NAMES):
        m = y_s == c
        ax.scatter(proj[m, 0], proj[m, 1],
                   s=5, alpha=0.35, c=PROFILE_COL[name],
                   label=f"{name} (n={m.sum()})",
                   linewidth=0)
    ax.set_xlabel(f"PC1 ({100*var_ratio[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100*var_ratio[1]:.0f}%)")
    ax.set_title(r"PCA of BeMAE embeddings, coloured by H$\alpha$ morphology")
    ax.legend(loc="upper right", frameon=True, fontsize=7, markerscale=2,
              framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_fig(fig, "fig3_embedding_pca")


if __name__ == "__main__":
    main()

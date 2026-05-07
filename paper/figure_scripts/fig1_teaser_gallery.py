"""Figure 1 (teaser) — Gallery of Hα profiles across the 5 morphologies.

Script Figure for : "BESS-Bench covers the full morphological
diversity of Be-star Hα profiles". We showcase one real spectrum per profile
class (absorption / single emission / double emission / shell / asymmetric
double).

Input
-----
``downstream/results/spectral_features.pt`` contains, for every star, the
normalized flux tensor ``flux_norm`` [N_spec, 128] together with the
``profile_type_id`` assigned by ``downstream/classify_profiles.py``.

Output
------
- ``figures/fig1_teaser_v2.pdf``
- ``figures/fig1_teaser_v2.png``

"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, PROFILE_COL, save_fig, setup_style  # noqa: E402

import os
FEATURES = Path(os.environ.get("BESS_FEATURES_PATH",
                               str(ROOT / "reference_results" / "features" / "spectral_features.pt")))

# Profile classes — IDs MUST match downstream/compute_spectral_features.py:
#   0 = absorption_pure
#   1 = single_peak
#   2 = double_peak
#   3 = double_asym
#   4 = shell
PROFILES: list[tuple[str, int, str]] = [
    ("absorption_pure", 0, "Pure absorption"),
    ("single_peak",     1, "Single-peak emission"),
    ("double_peak",     2, "Double-peak emission"),
    ("double_asym",     3, "Asymmetric V/R"),
    ("shell",           4, "Shell"),
]

# ── Per-class physics-based selection scores ─────────────────────────────────
# Every class gets a dedicated score that rewards the most *pedagogically clear*
# example.  All scores are fully algorithmic: no star name is hard-coded.
#
# Grid geometry (128 bins, [6530, 6595] Å):
#   bin ≈ 0.51 Å, H-alpha at bin ~63
#   _CORE window   : [59, 68)  — the H-alpha core (±4 bins ≈ ±2 Å)
#   _BLUE window   : [5,  58)  — blue wing, clear of the core
#   _RED  window   : [69, 123) — red wing, clear of the core
# Windows are kept away from the spectrum edges (first/last 5 bins) to avoid
# continuum edge artefacts.
_LAM_TMP    = np.linspace(6530.0, 6595.0, 128)
_HALPHA_BIN = int(np.argmin(np.abs(_LAM_TMP - 6562.8)))   # ~63
_CORE_LO, _CORE_HI = _HALPHA_BIN - 4, _HALPHA_BIN + 5    # [59, 68)
_BLUE_LO,  _BLUE_HI  = 5, _HALPHA_BIN - 5                # [5,  58)
_RED_LO,   _RED_HI   = _HALPHA_BIN + 6, 123              # [69, 123)
_EMIT_THRESH = 1.05


def absorption_score(flux: np.ndarray) -> float:
    """Pure absorption: deep, sharp trough at H-alpha, continuum near 1.0.

    Score = depth × sharpness.
    - depth    = 1 - core_min       : how far below continuum the trough dips.
    - sharpness = depth / wing_depth : trough must be localized at H-alpha
      (wings near 1.0), not a broad stellar absorption across the full range.
    Returns -1 if any emission is present or the trough is shallower than 3%.
    """
    if float(flux.max()) > _EMIT_THRESH:
        return -1.0
    core_min = float(flux[_CORE_LO:_CORE_HI].min())
    depth = 1.0 - core_min
    if depth < 0.03:          # require at least 3% absorption depth
        return -1.0
    blue_min  = float(flux[_BLUE_LO:_BLUE_HI].min())
    red_min   = float(flux[_RED_LO:_RED_HI].min())
    wing_depth = max(1.0 - min(blue_min, red_min), 1e-6)
    sharpness  = depth / wing_depth   # high when trough is much deeper than wings
    return depth * sharpness


def single_peak_score(flux: np.ndarray) -> float:
    """Single-peak emission: tall peak centred at Hα, dominant over wings.

    Score = log(1 + core_peak) × centre_dominance × smoothness.
    - log(1 + h): rewards height without extreme outlier domination.
    - centre_dominance = core_peak / max(blue_max, red_max): penalises
      profiles where side lobes rival the centre (would be double/asym).
    - smoothness = 1 - std/peak in the core: penalises noisy/spiky peaks.
    """
    core_peak = float(flux[_CORE_LO:_CORE_HI].max())
    if core_peak < _EMIT_THRESH:
        return -1.0
    blue_max = float(flux[_BLUE_LO:_BLUE_HI].max())
    red_max  = float(flux[_RED_LO:_RED_HI].max())
    side_max = max(blue_max, red_max, 1e-6)
    centre_dominance = core_peak / side_max
    smoothness = max(1.0 - float(flux[_CORE_LO:_CORE_HI].std()) / core_peak, 0.0)
    return float(np.log1p(core_peak)) * centre_dominance * smoothness


def double_peak_score(flux: np.ndarray) -> float:
    """Double-peak emission: two symmetric peaks, inter-peak dip ABOVE continuum.

    Key constraint: core_min >= 0.98 — rejects shell-like profiles where the
    central reversal dips below the continuum.

    Score = mean(V, R) × symmetry × sep_clarity × snr_proxy.
    sep_clarity = (mean(V,R) - core_max) / (mean(V,R) - 1):
        fraction of the emission height that is "carved" by the central dip.
        A true double-peak has this > 0.3; if the two peaks barely separate
        (plateau) it is close to 0.
    snr_proxy = (mean(V,R) - 1) / std(continuum): penalises noisy spectra.
    """
    v_peak = float(flux[_BLUE_LO:_BLUE_HI].max())
    r_peak = float(flux[_RED_LO:_RED_HI].max())
    if v_peak < _EMIT_THRESH or r_peak < _EMIT_THRESH:
        return -1.0
    # Reject shell-like profiles: inter-peak dip must stay above the continuum
    core_min = float(flux[_CORE_LO:_CORE_HI].min())
    if core_min < 0.98:
        return -1.0
    mean_height  = (v_peak + r_peak) / 2.0
    symmetry     = min(v_peak, r_peak) / max(v_peak, r_peak)
    core_max     = float(flux[_CORE_LO:_CORE_HI].max())
    emission_rng = max(mean_height - 1.0, 1e-6)
    sep_clarity  = max((mean_height - core_max) / emission_rng, 0.0)
    # Require a visible valley: the central dip must remove at least 25% of
    # the emission range (reject flat-top / barely-separated profiles).
    if sep_clarity < 0.25:
        return -1.0
    cont_std     = float(flux[105:120].std()) + 1e-6
    snr_proxy    = (mean_height - 1.0) / cont_std
    # Continuum flatness: far blue [5:20] and far red [108:123] should sit at ~1.0
    # Hard filter: reject if either wing drifts more than 1.2% from continuum.
    blue_cont_dev = abs(float(flux[5:20].mean())   - 1.0)
    red_cont_dev  = abs(float(flux[108:123].mean()) - 1.0)
    if blue_cont_dev > 0.012 or red_cont_dev > 0.012:
        return -1.0
    cont_quality  = max(1.0 - 10.0 * (blue_cont_dev + red_cont_dev), 0.0)
    return mean_height * symmetry * sep_clarity * snr_proxy * cont_quality


def double_asym_score(flux: np.ndarray) -> float:
    """Asymmetric V/R: two peaks clearly present but with different heights.

    Score = mean(V, R) × asymmetry, where asymmetry = |V-R|/(V+R).
    High score ↔ both peaks are strong AND visibly unequal.
    """
    v_peak = float(flux[_BLUE_LO:_BLUE_HI].max())
    r_peak = float(flux[_RED_LO:_RED_HI].max())
    if v_peak < _EMIT_THRESH or r_peak < _EMIT_THRESH:
        return -1.0
    mean_height = (v_peak + r_peak) / 2.0
    asymmetry   = abs(v_peak - r_peak) / (v_peak + r_peak)
    return mean_height * asymmetry


def shell_score(flux: np.ndarray) -> float:
    """Shell: deep central absorption flanked by two emission peaks.

    Score = (1 - flux_min_core) × bonus_two_peaks.
    """
    depth  = 1.0 - float(flux[_CORE_LO:_CORE_HI].min())
    v_peak = float(flux[_BLUE_LO:_BLUE_HI].max())
    r_peak = float(flux[_RED_LO:_RED_HI].max())
    bonus  = 1.5 if (v_peak > _EMIT_THRESH and r_peak > _EMIT_THRESH) else 1.0
    return depth * bonus


_SCORE_FN = {
    "absorption_pure": absorption_score,
    "single_peak":     single_peak_score,
    "double_peak":     double_peak_score,
    "double_asym":     double_asym_score,
    "shell":           shell_score,
}

# Wavelength grid: 128 bins over [6530, 6595] Å  (see pipeline/04_export).
LAM = np.linspace(6530.0, 6595.0, 128)
HALPHA = 6562.8


def pick_examples(feats: dict) -> dict[str, tuple[str, int, np.ndarray]]:
    """Pick one representative spectrum per profile class.

    Every class uses a dedicated physics-based score (see ``_SCORE_FN``) that
    rewards the most pedagogically clear exemplar — the spectrum whose shape
    best illustrates the defining feature of that morphology.  No star name is
    hard-coded.
    """
    bucket: dict[int, list[tuple[str, int, np.ndarray]]] = {p: [] for _, p, _ in PROFILES}
    for star, d in feats.items():
        flux = d["flux_norm"].numpy()
        pids = d["profile_type_id"].numpy()
        for i in range(flux.shape[0]):
            p = int(pids[i])
            if p in bucket and not np.isnan(flux[i]).any():
                bucket[p].append((star, i, flux[i]))

    chosen: dict[str, tuple[str, int, np.ndarray]] = {}
    for name, pid, _ in PROFILES:
        items = bucket[pid]
        if not items:
            continue
        score_fn = _SCORE_FN[name]
        scores = np.array([score_fn(it[2]) for it in items])
        best = int(np.argmax(scores))
        chosen[name] = items[best]
        print(f"  [{name:16s}] n_class = {len(items):>6,d}  "
              f"picked star = {chosen[name][0]}  spec_idx = {chosen[name][1]}  "
              f"score = {scores[best]:.3f}")
    return chosen


def main() -> None:
    setup_style()
    feats = torch.load(FEATURES, map_location="cpu", weights_only=False)
    examples = pick_examples(feats)

    n = len(PROFILES)
    fig, axes = plt.subplots(1, n, figsize=(9.6, 2.4), sharey=False,
                             gridspec_kw={"wspace": 0.18})
    for ax, (name, _pid, pretty) in zip(axes, PROFILES):
        ax.axvline(HALPHA, color="0.8", lw=0.6, zorder=0)
        ax.axhline(1.0, color="0.85", lw=0.5, zorder=0)

        if name not in examples:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        star, idx, flux = examples[name]
        ax.plot(LAM, flux, color=PROFILE_COL[name], lw=1.0)
        ax.set_title(pretty, color=PROFILE_COL[name], fontsize=9)
        # Place star label inside lower-right corner to avoid title overlap
        ax.text(0.97, 0.05, f"{star}", transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom", color="0.35")
        ax.set_xlim(LAM.min(), LAM.max())
        ax.set_xticks([6540, 6562.8, 6585])
        ax.set_xticklabels(["6540", r"H$\alpha$", "6585"])  # mathtext
        # Per-panel y limits: 10% padding around [flux_min, flux_max] so that
        # every morphology is maximally legible at its own dynamic range.
        flo, fhi = float(flux.min()), float(flux.max())
        pad = max((fhi - flo) * 0.12, 0.05)
        ax.set_ylim(flo - pad, fhi + pad)

        ax.tick_params(length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Normalised flux", fontsize=9)
    fig.supxlabel(r"Wavelength ($\AA$)", y=-0.02, fontsize=9)
    fig.suptitle(r"Five H$\alpha$ morphologies covered by BESS-Bench",
                 fontsize=10, y=1.03)
    save_fig(fig, "fig1_teaser_v2")


if __name__ == "__main__":
    main()

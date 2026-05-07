"""Sanity tests for compute_fwhm and compute_central_depth after the Phase 2 fix.

Synthetic profiles on the 128-bin Hα grid, no dataset needed.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from downstream.compute_spectral_features import (  # noqa: E402
    TARGET_GRID, HALPHA_CENTER, DELTA_LAMBDA,
    compute_fwhm, compute_central_depth, compute_ew,
)


def gaussian(wl, center, sigma, amp):
    return amp * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def test_single_peak_emission():
    # Single peak at Hα, sigma=5 Å → expected FWHM = 2*sqrt(2 ln 2) * 5 ≈ 11.77 Å
    f = 1.0 + gaussian(TARGET_GRID, HALPHA_CENTER, 5.0, 1.0)
    fw = compute_fwhm(f, TARGET_GRID)
    expected = 2 * np.sqrt(2 * np.log(2)) * 5.0
    print(f"[single emission] FWHM={fw:.2f} Å (expected ≈ {expected:.2f})")
    assert abs(fw - expected) < 2.0, fw


def test_symmetric_double_peak():
    # V peak at -4 Å, R peak at +4 Å, both amp=1.0, sigma=2 Å
    v = gaussian(TARGET_GRID, HALPHA_CENTER - 4, 2.0, 1.0)
    r = gaussian(TARGET_GRID, HALPHA_CENTER + 4, 2.0, 1.0)
    f = 1.0 + v + r
    fw = compute_fwhm(f, TARGET_GRID)
    # Enveloppe: from V blue edge to R red edge ~ 4+sigma*fwhm/2 each side
    print(f"[symmetric double] FWHM={fw:.2f} Å (expected ~10–12 Å)")
    assert 8 < fw < 15, fw


def test_asymmetric_double_peak_V_dominant():
    # V=2.5 above continuum, R=0.4 above continuum. Old code used max peak → lost R.
    v = gaussian(TARGET_GRID, HALPHA_CENTER - 4, 2.0, 2.5)
    r = gaussian(TARGET_GRID, HALPHA_CENTER + 4, 2.0, 0.4)
    f = 1.0 + v + r
    fw = compute_fwhm(f, TARGET_GRID)
    ew = compute_ew(f)
    print(f"[asym V>>R]       FWHM={fw:.2f} Å, EW={ew:.2f} (must reach R)")
    # Old buggy behaviour would give ~5 Å (V peak alone). New behaviour
    # uses min(peak_V, peak_R) as reference => half_level=1.2, mask reaches R.
    assert fw > 7, f"FWHM {fw:.2f} too narrow; R peak not captured"


def test_outlier_cosmic_beyond_core():
    # Symmetric double + a spurious cosmic spike at +20 Å.
    v = gaussian(TARGET_GRID, HALPHA_CENTER - 4, 2.0, 1.0)
    r = gaussian(TARGET_GRID, HALPHA_CENTER + 4, 2.0, 1.0)
    cosmic = gaussian(TARGET_GRID, HALPHA_CENTER + 20, 0.5, 3.0)
    f = 1.0 + v + r + cosmic
    fw = compute_fwhm(f, TARGET_GRID)
    print(f"[cosmic at +20Å]  FWHM={fw:.2f} Å (must stay <20 Å)")
    # Old buggy behaviour: last True index at +20 Å → FWHM ~24 Å.
    assert fw < 18, f"FWHM {fw:.2f} inflated by cosmic outside core"


def test_pure_absorption():
    f = 1.0 - gaussian(TARGET_GRID, HALPHA_CENTER, 4.0, 0.6)
    fw = compute_fwhm(f, TARGET_GRID)
    expected = 2 * np.sqrt(2 * np.log(2)) * 4.0
    print(f"[absorption]      FWHM={fw:.2f} Å (expected ≈ {expected:.2f})")
    assert abs(fw - expected) < 2.0, fw


def test_shell_profile():
    # V and R peaks + deep central dip: min around 0.3
    v = gaussian(TARGET_GRID, HALPHA_CENTER - 4, 2.0, 1.2)
    r = gaussian(TARGET_GRID, HALPHA_CENTER + 4, 2.0, 1.2)
    dip = gaussian(TARGET_GRID, HALPHA_CENTER, 1.5, 0.7)
    f = 1.0 + v + r - dip
    depth = compute_central_depth(f, TARGET_GRID)
    fw = compute_fwhm(f, TARGET_GRID)
    print(f"[shell]           depth={depth:.3f}, FWHM={fw:.2f} Å")
    assert depth > 0.2, f"shell depth {depth:.3f} too small"
    # Expected span from V blue edge to R red edge, ~10 Å
    assert 7 < fw < 20, fw


def test_depth_window_3A_catches_dip():
    # Narrow shell dip on a normalised continuum (F=1) with emission peaks
    # at ±4 Å and a 2 Å-wide central absorption slightly offset by +0.5 Å
    # (mimicking ~1-bin shift from uncorrected heliocentric frame).
    v = gaussian(TARGET_GRID, HALPHA_CENTER - 4, 2.0, 1.2)
    r = gaussian(TARGET_GRID, HALPHA_CENTER + 4, 2.0, 1.2)
    dip = gaussian(TARGET_GRID, HALPHA_CENTER + 0.5, 0.8, 1.5)  # offset 0.5 Å
    f = 1.0 + v + r - dip
    depth = compute_central_depth(f, TARGET_GRID)
    print(f"[narrow dip @+0.5Å] depth={depth:.3f} (±3 Å window must catch it)")
    # The old ±2 Å window (2.56 bins) could still catch it; the ±3 Å window
    # is strictly more robust to sub-bin centring errors. Check depth > 0.3.
    assert depth > 0.3


if __name__ == "__main__":
    for fn in [
        test_single_peak_emission,
        test_symmetric_double_peak,
        test_asymmetric_double_peak_V_dominant,
        test_outlier_cosmic_beyond_core,
        test_pure_absorption,
        test_shell_profile,
        test_depth_window_3A_catches_dip,
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            raise
    print("\nAll synthetic tests passed.")

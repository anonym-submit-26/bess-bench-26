"""
compute_spectral_features.py — Compute Hα spectral features for all BeSS spectra.

For each BeSS spectrum covering Hα ±50 Å, computes:
  - EW         : Equivalent width (integral)
  - FWHM       : Full width at half maximum of the profile
  - V/R        : Ratio of violet / red peak intensities
  - Δv         : Peak separation in km/s
  - depth      : Central absorption depth
  - peak_I     : Maximum profile intensity (emission peak)
  - profile_type : Morphological classification (absorption / single / double / shell)

Features are saved in a .pt file organized by (star, mjd) to align with
precomputed embeddings in star_data.pt.

Usage:
    python compute_spectral_features.py
    python compute_spectral_features.py --max_spectra 1000  # quick test
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

# ── Constantes ───────────────────────────────────────────────────────────────

HALPHA_CENTER = 6562.8       # Å
HALPHA_HALF_WINDOW = 50.0    # Å
N_BINS = 128
LAMBDA_MIN = HALPHA_CENTER - HALPHA_HALF_WINDOW   # 6512.8
LAMBDA_MAX = HALPHA_CENTER + HALPHA_HALF_WINDOW    # 6612.8
TARGET_GRID = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_BINS)
DELTA_LAMBDA = TARGET_GRID[1] - TARGET_GRID[0]     # ~0.78 Å

# Speed of light for V/R km/s conversion
C_KMS = 299792.458


# ══════════════════════════════════════════════════════════════════════════════
# CROP + NORMALIZATION (identical to stage1_encoder/dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

def covers_halpha_window(wl_min: float, wl_max: float) -> bool:
    return wl_min <= LAMBDA_MIN and wl_max >= LAMBDA_MAX


def crop_and_interpolate(flux, wavelength, target_grid=TARGET_GRID,
                        rqvh_kms=None):
    """Crop + interpolation + heliocentric velocity correction.

    Applies λ_corr = λ_obs × (1 - rqvh/c) where rqvh = BSS_RQVH (km/s),
    i.e. the residual heliocentric velocity not applied in the FITS header.
    Identical to stage1_encoder/dataset.py pipeline for consistency
    between target features and embeddings.
    """
    if rqvh_kms is not None and np.isfinite(rqvh_kms) and rqvh_kms != 0.0:
        wavelength = wavelength * (1.0 - rqvh_kms / C_KMS)
    margin = 2.0
    mask = (wavelength >= LAMBDA_MIN - margin) & (wavelength <= LAMBDA_MAX + margin)
    if mask.sum() < 5:
        return None
    wl_crop = wavelength[mask]
    fl_crop = flux[mask]
    sort_idx = np.argsort(wl_crop)
    wl_crop = wl_crop[sort_idx]
    fl_crop = fl_crop[sort_idx]
    return np.interp(target_grid, wl_crop, fl_crop).astype(np.float32)


def normalize_local_continuum(flux, n_edge=10):
    if len(flux) < 2 * n_edge:
        n_edge = max(1, len(flux) // 4)
    left = flux[:n_edge]
    right = flux[-n_edge:]
    continuum = np.mean(np.concatenate([left, right]))
    if continuum <= 0 or not np.isfinite(continuum):
        continuum = 1.0
    normalized = flux / continuum
    normalized = np.clip(normalized, -5.0, 10.0)
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=10.0, neginf=-5.0)
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ew(flux_norm, delta_lambda=DELTA_LAMBDA):
    """
    EW = ∫ (1 - F(λ)) dλ over the central ±30 Å window.
    EW > 0 = absorption, EW < 0 = emission.
    """
    center_idx = len(flux_norm) // 2
    half_window = int(30.0 / delta_lambda)  # ~38 bins
    i_start = max(0, center_idx - half_window)
    i_end = min(len(flux_norm), center_idx + half_window)
    flux_window = flux_norm[i_start:i_end]
    return float(np.sum((1.0 - flux_window)) * delta_lambda)


def compute_fwhm(flux_norm, wavelengths=TARGET_GRID):
    """
    FWHM of the central H$\\alpha$ feature (emission envelope or absorption).

    Conventions
    -----------
    * Emission envelope (Hanuschik 1996 / Silaj 2010 standard):
        half_level = 1 + (min(peak_V, peak_R) - 1) / 2
      i.e. we measure the width of the emission complex at half of the
      *smaller* peak. This reaches both V and R even when strongly asymmetric
      and degrades gracefully to single-peak when only one peak is found.
    * Absorption: half_level = (1 + min) / 2, measured below the continuum.
    * Contiguity: we only keep the cluster of pixels that contains the
      central bin, so a cosmic/contaminant outside the Hα core cannot
      artificially widen the FWHM.
    """
    center_idx = len(flux_norm) // 2

    # Search window ±25 Å around Hα
    half_w = int(25.0 / DELTA_LAMBDA)
    i_lo = max(0, center_idx - half_w)
    i_hi = min(len(flux_norm), center_idx + half_w)
    f_center = flux_norm[i_lo:i_hi]
    wl_center = wavelengths[i_lo:i_hi]
    center_local = center_idx - i_lo

    # Decide emission vs absorption by profile shape, not by EW threshold:
    # a local extremum > 1.05 (resp. < 0.95) within ±5 Å around Hα.
    close = int(5.0 / DELTA_LAMBDA)
    core = f_center[max(0, center_local - close): center_local + close + 1]
    is_emission = (float(np.max(core)) - 1.0) > (1.0 - float(np.min(core)))

    if is_emission:
        # For emission, pick the half-level from the *smaller* of the V and R
        # peaks to avoid missing the weaker wing on strongly asymmetric profiles.
        left_seg = f_center[:center_local]
        right_seg = f_center[center_local:]
        peak_left = float(np.max(left_seg)) if len(left_seg) else 1.0
        peak_right = float(np.max(right_seg)) if len(right_seg) else 1.0
        # Fallback if no true double peak: use the single maximum.
        # Threshold aligned with find_emission_peaks / classify_profile (1.05)
        # for end-to-end consistency between FWHM, V/R and morphology class.
        if peak_left <= 1.05 or peak_right <= 1.05:
            peak_ref = max(peak_left, peak_right)
        else:
            peak_ref = min(peak_left, peak_right)
        if peak_ref <= 1.0:
            return float("nan")
        half_level = 1.0 + (peak_ref - 1.0) / 2.0
        mask = f_center >= half_level
    else:
        min_val = float(np.min(f_center))
        if min_val >= 1.0:
            return float("nan")
        half_level = (1.0 + min_val) / 2.0
        mask = f_center <= half_level

    if not mask.any():
        return float("nan")

    # Restrict to the contiguous cluster that contains the centre bin.
    # If the centre bin itself is not in the mask (e.g. shell central dip
    # for an emission profile), we take the union of the two clusters
    # adjacent to the centre — this preserves Hanuschik's convention of
    # reporting the full emission envelope V_edge → R_edge.
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return float("nan")

    if mask[center_local]:
        # Contiguous cluster around the centre.
        left = center_local
        while left > 0 and mask[left - 1]:
            left -= 1
        right = center_local
        while right < len(mask) - 1 and mask[right + 1]:
            right += 1
    else:
        # Centre bin below half_level (shell case for emission): union of
        # the last True-cluster on the left of centre and the first on the
        # right of centre.
        left_candidates = idx[idx < center_local]
        right_candidates = idx[idx > center_local]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            return float("nan")
        # Left edge = start of the last cluster before centre
        left = int(left_candidates[-1])
        while left > 0 and mask[left - 1]:
            left -= 1
        # Right edge = end of the first cluster after centre
        right = int(right_candidates[0])
        while right < len(mask) - 1 and mask[right + 1]:
            right += 1

    fwhm = float(wl_center[right] - wl_center[left])
    return max(fwhm, 0.0)


def find_emission_peaks(flux_norm, wavelengths=TARGET_GRID):
    """
    Detect emission peaks in the Hα ±20 Å window.

    Returns a list of dicts: [{idx, wavelength, intensity}, ...]
    sorted by decreasing intensity.
    """
    center_idx = len(flux_norm) // 2
    half_w = int(20.0 / DELTA_LAMBDA)  # ±20 Å
    i_lo = max(0, center_idx - half_w)
    i_hi = min(len(flux_norm), center_idx + half_w)

    f_center = flux_norm[i_lo:i_hi]
    wl_center = wavelengths[i_lo:i_hi]

    # Light smoothing to reduce noise
    if len(f_center) > 7:
        f_smooth = savgol_filter(f_center, window_length=5, polyorder=2)
    else:
        f_smooth = f_center

    # Detect peaks above the continuum.
    # Tightened thresholds (audit 2026/05): 1.02/0.02 let noise through at
    # the foot of the main peak, corrupting V/R on ~30% of `double_asym`.
    peak_indices, properties = find_peaks(
        f_smooth,
        height=1.05,            # at least 5% above continuum
        distance=3,             # at least 3 bins (~2.3 Å) between two peaks
        prominence=0.04,        # minimum prominence (anti-noise)
    )

    peaks = []
    for idx in peak_indices:
        peaks.append({
            "idx": int(idx + i_lo),
            "wavelength": float(wl_center[idx]),
            "intensity": float(f_center[idx]),   # unsmoothed intensity
        })

    # Sort by decreasing intensity
    peaks.sort(key=lambda p: -p["intensity"])
    return peaks


# V/R guard rails to prevents spurious V/R ratios (>>10) when one
# peak is actually a marginal false positive at the foot of the dominant peak.
_VR_MIN_PEAK_INTENSITY = 1.05   # both peaks must rise ≥ 5% above continuum
_VR_MIN_SEPARATION_A   = 2.0    # both peaks must be separated by ≥ 2 Å
# Physical V/R range (Hanuschik 1996, Silaj 2010) is [0.3, 3]; values
# beyond [0.2, 5] indicate a shoulder mistakenly tagged as a true peak
# rather than a real V/R asymmetry — we mark them invalid.
_VR_MAX_RATIO = 5.0
_VR_MIN_RATIO = 0.2


def compute_vr_ratio(flux_norm, wavelengths=TARGET_GRID):
    """
    V/R ratio: violet peak / red peak of the Hα emission profile.

    V = emission peak at λ < Hα (blue/violet side)
    R = emission peak at λ > Hα (red side)

    Returns: V/R ratio (float), NaN if no real double-peak detected (both
    peaks must exceed 1.05 and be separated by at least 2 Å to qualify).
    """
    peaks = find_emission_peaks(flux_norm, wavelengths)

    if len(peaks) < 2:
        return float("nan")

    v_peaks = [p for p in peaks if p["wavelength"] < HALPHA_CENTER]
    r_peaks = [p for p in peaks if p["wavelength"] > HALPHA_CENTER]

    if not v_peaks or not r_peaks:
        return float("nan")

    v_peak = max(v_peaks, key=lambda p: p["intensity"])
    r_peak = max(r_peaks, key=lambda p: p["intensity"])

    # Guard rails (anti-false-peak)
    if (v_peak["intensity"] < _VR_MIN_PEAK_INTENSITY
            or r_peak["intensity"] < _VR_MIN_PEAK_INTENSITY):
        return float("nan")
    if (r_peak["wavelength"] - v_peak["wavelength"]) < _VR_MIN_SEPARATION_A:
        return float("nan")

    vr = (v_peak["intensity"] - 1.0) / (r_peak["intensity"] - 1.0)
    # Outside physical range → one peak is most likely a shoulder or
    # artefact at the foot of a dominant peak.
    if not (_VR_MIN_RATIO <= vr <= _VR_MAX_RATIO):
        return float("nan")
    return float(vr)


def compute_peak_separation(flux_norm, wavelengths=TARGET_GRID):
    """
    Δv: separation in km/s between the two emission peaks.
    Δv = c × (λ_R - λ_V) / Hα

    Returns: Δv in km/s (float), NaN if no double-peak.
    """
    peaks = find_emission_peaks(flux_norm, wavelengths)

    if len(peaks) < 2:
        return float("nan")

    v_peaks = [p for p in peaks if p["wavelength"] < HALPHA_CENTER]
    r_peaks = [p for p in peaks if p["wavelength"] > HALPHA_CENTER]

    if not v_peaks or not r_peaks:
        return float("nan")

    v_peak = max(v_peaks, key=lambda p: p["intensity"])
    r_peak = max(r_peaks, key=lambda p: p["intensity"])

    delta_lambda = r_peak["wavelength"] - v_peak["wavelength"]
    delta_v = C_KMS * delta_lambda / HALPHA_CENTER

    return float(delta_v)


def compute_central_depth(flux_norm, wavelengths=TARGET_GRID):
    """
    Central absorption depth between the two emission peaks.

    depth = 1 - F(Hα_center) / F(continuum)

    Shell profile: depth > 0.3 (strong absorption in center of emission).
    Simple double-peak: depth ~ 0.05-0.2.
    Single-peak: depth ≈ 0 or negative.
    """
    # Minimum flux in the central ±3 Å window: covers the shell trough
    # (typically 1–2 Å wide) plus margin for profiles slightly shifted
    # by imperfect heliocentric correction (~1 bin).
    center_half = int(3.0 / DELTA_LAMBDA)
    center_idx = len(flux_norm) // 2
    i_lo = max(0, center_idx - center_half)
    i_hi = min(len(flux_norm), center_idx + center_half + 1)

    f_at_center = np.min(flux_norm[i_lo:i_hi])

    # If in emission, depth is relative to the emission peak
    peaks = find_emission_peaks(flux_norm, wavelengths)
    if len(peaks) >= 1:
        max_peak = peaks[0]["intensity"]
        if max_peak > 1.05:
            # depth relative to the emission peak
            depth = 1.0 - f_at_center / max_peak
            return float(depth)

    # If absorption: depth relative to the continuum
    depth = 1.0 - f_at_center
    return float(depth)


def compute_peak_intensity(flux_norm, wavelengths=TARGET_GRID):
    """
    Maximum profile intensity in the Hα ±15 Å window.
    I_peak / I_continuum.
    """
    center_idx = len(flux_norm) // 2
    half_w = int(15.0 / DELTA_LAMBDA)
    i_lo = max(0, center_idx - half_w)
    i_hi = min(len(flux_norm), center_idx + half_w)
    return float(np.max(flux_norm[i_lo:i_hi]))


def classify_profile(flux_norm, wavelengths=TARGET_GRID, ew=None):
    """
    Morphological classification of the Hα profile:
      0 = absorption_pure   (no disk, F < 1 at center)
      1 = single_peak       (single-peak emission, pole-on disk)
      2 = double_peak       (double-peak emission, V≈R)
      3 = double_asym       (asymmetric double-peak, V≠R)
      4 = shell             (emission + deep central absorption)

    Returns: (int class_id, str class_name)
    """
    if ew is None:
        ew = compute_ew(flux_norm)

    peaks = find_emission_peaks(flux_norm, wavelengths)
    depth = compute_central_depth(flux_norm, wavelengths)

    # No significant emission - pure absorption
    if ew > -0.5 or len(peaks) == 0:
        return 0, "absorption_pure"

    # Check for real V and R peaks
    # (same guard rails as compute_vr_ratio: intensity ≥ 1.05, separation ≥ 2 Å)
    v_peaks = [p for p in peaks
               if p["wavelength"] < HALPHA_CENTER - 1.0
               and p["intensity"] >= _VR_MIN_PEAK_INTENSITY]
    r_peaks = [p for p in peaks
               if p["wavelength"] > HALPHA_CENTER + 1.0
               and p["intensity"] >= _VR_MIN_PEAK_INTENSITY]
    has_double = False
    if v_peaks and r_peaks:
        v_peak = max(v_peaks, key=lambda p: p["intensity"])
        r_peak = max(r_peaks, key=lambda p: p["intensity"])
        has_double = (r_peak["wavelength"] - v_peak["wavelength"]) >= _VR_MIN_SEPARATION_A

    if has_double:
        # Shell: double-peak + deep central absorption
        if depth > 0.3:
            return 4, "shell"

        # Confirmed double-peak with V/R in physical range [0.2, 5]
        # (otherwise the "second peak" is a noisy shoulder, not a real peak)
        vr = compute_vr_ratio(flux_norm, wavelengths)
        if np.isnan(vr):
            return 1, "single_peak"
        if vr < 0.8 or vr > 1.2:
            return 3, "double_asym"
        return 2, "double_peak"

    # Single peak emission
    return 1, "single_peak"


def is_saturated_spectrum(flux_norm) -> bool:
    """Spectra whose normalization has diverged (clipping (-5, +10) active).

    Empirical criterion : peak > 5× continuum or min < -1
    indicates a continuum problem (parasitic lines, dead pixels, division
    by weak flux). These spectra are kept in the HF dataset (raw FITS)
    but flagged for downstream tasks.
    """
    return bool(np.max(flux_norm) > 5.0 or np.min(flux_norm) < -1.0)


def compute_all_features(flux_norm, wavelengths=TARGET_GRID):
    """
    Compute all spectral features for a normalized spectrum.

    Returns a dict with all features.
    """
    ew = compute_ew(flux_norm)
    fwhm = compute_fwhm(flux_norm, wavelengths)
    vr = compute_vr_ratio(flux_norm, wavelengths)
    delta_v = compute_peak_separation(flux_norm, wavelengths)
    depth = compute_central_depth(flux_norm, wavelengths)
    peak_i = compute_peak_intensity(flux_norm, wavelengths)
    profile_id, profile_name = classify_profile(flux_norm, wavelengths, ew=ew)
    saturated = is_saturated_spectrum(flux_norm)

    return {
        "ew": ew,
        "fwhm": fwhm,
        "vr_ratio": vr,
        "delta_v": delta_v,
        "central_depth": depth,
        "peak_intensity": peak_i,
        "profile_type_id": profile_id,
        "profile_type_name": profile_name,
        "is_saturated": saturated,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute spectral features for all Hα spectra"
    )
    parser.add_argument("--max_spectra", type=int, default=None,
                        help="Max spectra to process (for testing)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "reference_results" / "features" / "spectral_features.pt"),
                        help="Output file path")
    parser.add_argument("--batch_log", type=int, default=5000,
                        help="Log every N spectra")
    parser.add_argument("--snr_min", type=float, default=10.0,
                        help="Reject spectra whose DER_SNR (Stoehr 2008) "
                             "is below this threshold. Default 10 matches "
                             "the quality floor announced in the paper §3.2.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load the HF dataset ──
    print("Loading BeSS dataset from HuggingFace...")
    from datasets import load_dataset  # noqa

    ds = load_dataset("anonym-submit-26/bess-bench-26", split="train")
    print(f"  Total raw: {len(ds):,} spectra")

    # ── Process spectra one by one ──
    # Features indexed by (star_name, mjd) to align with star_data.pt later.
    features_by_star = defaultdict(lambda: {
        "mjds": [],
        "ew": [],
        "fwhm": [],
        "vr_ratio": [],
        "delta_v": [],
        "central_depth": [],
        "peak_intensity": [],
        "profile_type_id": [],
        "is_saturated": [],
        "flux_norm": [],           # normalized flux kept for downstream use
    })

    n_processed = 0
    n_skipped = 0
    n_low_snr = 0
    profile_counts = defaultdict(int)

    for i in range(len(ds)):
        if args.max_spectra and n_processed >= args.max_spectra:
            break

        row = ds[i]
        flux_raw = np.array(row["flux"], dtype=np.float32)
        wavelength = np.array(row["wavelength"], dtype=np.float32)
        star_name = row.get("star_name", "")
        mjd = float(row.get("mjd", 0.0))
        snr = row.get("snr", None)

        # BSS_RQVH (residual heliocentric velocity, km/s)
        rqvh_raw = row.get("bss_rqvh", None)
        rqvh_kms = float(rqvh_raw) if rqvh_raw is not None else 0.0
        if not np.isfinite(rqvh_kms):
            rqvh_kms = 0.0

        if not star_name or mjd <= 0:
            n_skipped += 1
            continue

        # SNR threshold: discard spectra below SNR_MIN (DER_SNR, Stoehr 2008).
        # This enforces the quality floor claimed in paper §3.2.
        if snr is not None and np.isfinite(snr) and snr < args.snr_min:
            n_low_snr += 1
            continue

        # Check Hα window coverage
        wl_min = wavelength.min() if len(wavelength) > 0 else 0
        wl_max = wavelength.max() if len(wavelength) > 0 else 0
        if not covers_halpha_window(wl_min, wl_max):
            n_skipped += 1
            continue

        # Crop and interpolate (with heliocentric correction)
        flux_interp = crop_and_interpolate(
            flux_raw, wavelength, rqvh_kms=rqvh_kms
        )
        if flux_interp is None:
            n_skipped += 1
            continue

        # Local continuum normalization
        flux_norm = normalize_local_continuum(flux_interp)

        # Compute all spectral features
        features = compute_all_features(flux_norm)

        # Store
        sd = features_by_star[star_name]
        sd["mjds"].append(mjd)
        sd["ew"].append(features["ew"])
        sd["fwhm"].append(features["fwhm"])
        sd["vr_ratio"].append(features["vr_ratio"])
        sd["delta_v"].append(features["delta_v"])
        sd["central_depth"].append(features["central_depth"])
        sd["peak_intensity"].append(features["peak_intensity"])
        sd["profile_type_id"].append(features["profile_type_id"])
        sd["is_saturated"].append(features["is_saturated"])
        sd["flux_norm"].append(flux_norm)

        profile_counts[features["profile_type_name"]] += 1
        n_processed += 1

        if n_processed % args.batch_log == 0:
            print(f"  {n_processed:,} spectra processed, "
                  f"{len(features_by_star)} stars, "
                  f"{n_skipped:,} skipped, "
                  f"{n_low_snr:,} rejected SNR<{args.snr_min}")

    print(f"\nDone: {n_processed:,} spectra, "
          f"{len(features_by_star)} stars, "
          f"{n_skipped:,} filtered (coverage/mjd/interp), "
          f"{n_low_snr:,} rejected SNR<{args.snr_min}")

    # ── Profile type distribution ──
    print("\nProfile type distribution:")
    for ptype, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(n_processed, 1)
        print(f"  {ptype:20s}: {count:6d} ({pct:5.1f}%)")

    # ── Convert to tensors, sort by MJD ──
    print("\nConverting to tensors and sorting by MJD...")
    final_data = {}

    for star, data in features_by_star.items():
        indices = np.argsort(data["mjds"])
        n_obs = len(indices)

        final_data[star] = {
            "mjds":            torch.tensor([data["mjds"][i] for i in indices], dtype=torch.float32),
            "ew":              torch.tensor([data["ew"][i] for i in indices], dtype=torch.float32),
            "fwhm":            torch.tensor([data["fwhm"][i] for i in indices], dtype=torch.float32),
            "vr_ratio":        torch.tensor([data["vr_ratio"][i] for i in indices], dtype=torch.float32),
            "delta_v":         torch.tensor([data["delta_v"][i] for i in indices], dtype=torch.float32),
            "central_depth":   torch.tensor([data["central_depth"][i] for i in indices], dtype=torch.float32),
            "peak_intensity":  torch.tensor([data["peak_intensity"][i] for i in indices], dtype=torch.float32),
            "profile_type_id": torch.tensor([data["profile_type_id"][i] for i in indices], dtype=torch.long),
            "is_saturated":    torch.tensor([data["is_saturated"][i] for i in indices], dtype=torch.bool),
            "flux_norm":       torch.tensor(
                np.stack([data["flux_norm"][i] for i in indices]), dtype=torch.float32
            ),  # [n_obs, 128]
        }

    # ── Save ──
    print(f"\nSaving to {output_path}...")
    torch.save(final_data, output_path)

    # ── Final stats ──
    total_spectra = sum(d["mjds"].shape[0] for d in final_data.values())
    n_finite_fwhm = sum(int(torch.isfinite(d["fwhm"]).sum()) for d in final_data.values())
    n_finite_vr = sum(int(torch.isfinite(d["vr_ratio"]).sum()) for d in final_data.values())
    n_finite_dv = sum(int(torch.isfinite(d["delta_v"]).sum()) for d in final_data.values())

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Stars           : {len(final_data)}")
    print(f"  Spectra         : {total_spectra}")
    print(f"  Valid FWHM      : {n_finite_fwhm} ({100*n_finite_fwhm/total_spectra:.1f}%)")
    print(f"  Valid V/R       : {n_finite_vr} ({100*n_finite_vr/total_spectra:.1f}%)")
    print(f"  Valid Δv        : {n_finite_dv} ({100*n_finite_dv/total_spectra:.1f}%)")
    print(f"  File            : {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Save stats too
    stats = {
        "n_stars": len(final_data),
        "n_spectra": total_spectra,
        "n_finite_fwhm": n_finite_fwhm,
        "n_finite_vr": n_finite_vr,
        "n_finite_dv": n_finite_dv,
        "n_rejected_low_snr": n_low_snr,
        "snr_min_threshold": args.snr_min,
        "profile_distribution": dict(profile_counts),
    }
    stats_path = output_path.parent / "spectral_features_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats           : {stats_path}")


if __name__ == "__main__":
    main()

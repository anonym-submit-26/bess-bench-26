"""
dataset.py — DataLoader for Hα-cropped spectra from HuggingFace.

Specialised data pipeline for the Hα encoder:

  1. Load from the HuggingFace Hub
  2. Filter: keep only spectra covering Hα ±50 Å
     (non-echelle AND echelle orders)
  3. Crop: extract the window [6512.8, 6612.8] Å
  4. Interpolation: re-grid onto 128 uniformly spaced bins
  5. Normalisation: divide by the local pseudo-continuum (edges)
  6. Star split (same test stars as the full encoder)

Result: each spectrum = vector of 128 floats centred on Hα,
with 100% of the signal physically relevant to Be phases.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from config import TrainConfig, HALPHA_CENTER, HALPHA_HALF_WINDOW, N_BINS


# ── Constants ─────────────────────────────────────────────────────────

# Crop window bounds
LAMBDA_MIN = HALPHA_CENTER - HALPHA_HALF_WINDOW  # 6512.8 Å
LAMBDA_MAX = HALPHA_CENTER + HALPHA_HALF_WINDOW  # 6612.8 Å

# Target interpolation grid (128 bins, δλ = 0.78125 Å)
TARGET_GRID = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_BINS)
DELTA_LAMBDA = TARGET_GRID[1] - TARGET_GRID[0]  # ~0.78 Å

# Speed of light (km/s) for heliocentric correction
C_KMS = 299792.458


# ── Helper functions ──────────────────────────────────────────────────────

def covers_halpha_window(wl_min: float, wl_max: float) -> bool:
    """
    Check whether a spectrum covers the full window [6512.8, 6612.8] Å.

    Stricter than the full encoder: we require COMPLETE coverage
    of the crop window, not just ±10 Å around the centre.
    """
    return wl_min <= LAMBDA_MIN and wl_max >= LAMBDA_MAX


def crop_and_interpolate(flux: np.ndarray, wavelength: np.ndarray,
                         target_grid: np.ndarray = TARGET_GRID,
                         rqvh_kms: float | None = None) -> np.ndarray:
    """
    Crop the spectrum to the Hα window and interpolate onto the target grid.

    Steps:
      1. Heliocentric correction: λ_corr = λ_obs × (1 - rqvh/c)
         where rqvh (BSS_RQVH) is the residual heliocentric velocity not
         applied in the original FITS (km/s). If None/NaN → rqvh=0.
         This guarantees cross-survey comparability (SDSS/LAMOST/APOGEE
         are all in the heliocentric frame).
      2. Select pixels in [6512.8, 6612.8] Å
      3. Linear interpolation onto the 128-bin grid
      4. Edges are extrapolated if necessary (fill_value=extrap)

    Returns a 128-float vector.
    """
    # 1. Heliocentric correction (before any masking)
    if rqvh_kms is not None and np.isfinite(rqvh_kms) and rqvh_kms != 0.0:
        wavelength = wavelength * (1.0 - rqvh_kms / C_KMS)

    # Mask to select the window (with a small margin for interpolation)
    margin = 2.0  # 2 Å margin for edge interpolation
    mask = (wavelength >= LAMBDA_MIN - margin) & (wavelength <= LAMBDA_MAX + margin)

    if mask.sum() < 5:
        # Not enough pixels → return zeros (will be filtered upstream)
        return np.zeros(len(target_grid), dtype=np.float32)

    wl_crop = wavelength[mask]
    fl_crop = flux[mask]

    # Sort by wavelength (some spectra are not ordered)
    sort_idx = np.argsort(wl_crop)
    wl_crop = wl_crop[sort_idx]
    fl_crop = fl_crop[sort_idx]

    # Linear interpolation onto the target grid
    flux_interp = np.interp(target_grid, wl_crop, fl_crop)

    return flux_interp.astype(np.float32)


def normalize_local_continuum(flux: np.ndarray, n_edge: int = 10) -> np.ndarray:
    """
    Normalise flux by the pseudo-continuum estimated at the window edges.

    Method:
      1. Take the first and last n_edge pixels
      2. Estimate continuum = mean of edges (far from Hα centre)
      3. Divide flux by this continuum

    The window edges (±50 Å from Hα) lie in the stellar continuum
    for almost all Be stars. This is a simple but robust estimate
    of the reference level.

    Advantage over global normalisation: no bias towards
    spectral type (the continuum is normalised to 1.0 for all stars).
    """
    if len(flux) < 2 * n_edge:
        n_edge = max(1, len(flux) // 4)

    # Continuum = mean of edges
    left = flux[:n_edge]
    right = flux[-n_edge:]
    continuum = np.mean(np.concatenate([left, right]))

    # Guard against division by zero
    if continuum <= 0 or not np.isfinite(continuum):
        continuum = 1.0

    normalized = flux / continuum

    # Clamp extreme values
    normalized = np.clip(normalized, -5.0, 10.0)

    # Replace NaN/Inf
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=10.0, neginf=-5.0)

    return normalized


# ── PyTorch Dataset ───────────────────────────────────────────────────────

class HalphaSpectralDataset(Dataset):
    """
    PyTorch Dataset for Hα spectra cropped to 128 bins.

    For each spectrum, returns:
      - flux       : normalised flux, 128 bins [128]
      - wavelengths: wavelength grid [128]
      - validity   : validity mask [128] (always 1 as interpolated)
      - star_name  : star name
      - mjd        : observation date (Modified Julian Date)
      - snr        : signal-to-noise ratio
      - ew         : Hα equivalent width (computed on raw flux)
      - is_echelle : boolean, True if echelle order
    """

    def __init__(
        self,
        hf_dataset,
        n_bins: int = N_BINS,
        norm_mode: str = "local_continuum",
        n_edge_pixels: int = 10,
    ):
        self.data = hf_dataset
        self.n_bins = n_bins
        self.norm_mode = norm_mode
        self.n_edge_pixels = n_edge_pixels
        # Pre-computed target grid
        self.target_grid = np.linspace(LAMBDA_MIN, LAMBDA_MAX, n_bins).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Load raw data
        flux_raw = np.array(row["flux"], dtype=np.float32)
        wavelength = np.array(row["wavelength"], dtype=np.float32)

        # Heliocentric correction: retrieve BSS_RQVH (km/s) if available
        rqvh_raw = row.get("bss_rqvh", None)
        rqvh_kms = float(rqvh_raw) if rqvh_raw is not None else 0.0
        if not np.isfinite(rqvh_kms):
            rqvh_kms = 0.0

        # Crop and interpolation onto the 128-bin Hα grid (with helio correction)
        flux_interp = crop_and_interpolate(
            flux_raw, wavelength, self.target_grid, rqvh_kms=rqvh_kms
        )

        # Normalisation by local continuum (window edges)
        if self.norm_mode == "local_continuum":
            flux_norm = normalize_local_continuum(flux_interp, self.n_edge_pixels)
        else:
            # Fallback zscore
            mu = flux_interp.mean()
            std = flux_interp.std()
            flux_norm = (flux_interp - mu) / max(std, 1e-10)
            flux_norm = np.clip(flux_norm, -5.0, 10.0)

        # EW computation directly on the normalised interpolated flux
        ew = self._compute_ew(flux_norm)

        return {
            "flux": torch.from_numpy(flux_norm),                    # [128]
            "wavelengths": torch.from_numpy(self.target_grid.copy()),  # [128]
            "validity": torch.ones(self.n_bins),                    # [128] always valid
            "star_name": row.get("star_name", ""),
            "mjd": float(row.get("mjd") or 0.0),
            "snr": float(row.get("snr") or 0.0),
            "ew": ew,
            "is_echelle": bool(row.get("is_echelle_order", False)),
            "length": self.n_bins,
        }

    def _compute_ew(self, flux_norm: np.ndarray) -> float:
        """
        Compute the Hα EW from the normalised flux on the 128-bin grid.

        The flux is already normalised by the continuum (≈ 1.0 at edges).
        EW = ∫ (1 - F(λ)) dλ

        Convention: EW > 0 = absorption, EW < 0 = emission.
        """
        # Central window ±30 Å for EW (not the full 100 Å window)
        center_idx = self.n_bins // 2
        half_ew_window = int(30.0 / DELTA_LAMBDA)  # ~38 bins
        i_start = max(0, center_idx - half_ew_window)
        i_end = min(self.n_bins, center_idx + half_ew_window)

        flux_window = flux_norm[i_start:i_end]
        ew = np.sum((1.0 - flux_window)) * DELTA_LAMBDA
        return float(ew)


# ── Filtering ───────────────────────────────────────────────────────────

def filter_halpha_spectra(hf_dataset, include_echelle: bool = True,
                          min_snr: float = None):
    """
    Filter the HF dataset to keep only spectra covering Hα ±50 Å.

    Criteria:
      - lambda_min <= 6512.8 Å AND lambda_max >= 6612.8 Å
      - Optional: include echelle orders (is_echelle_order)
      - Optional: filter by minimum SNR
    """
    def keep(row):
        wl_min = row.get("lambda_min", 0) or 0
        wl_max = row.get("lambda_max", 0) or 0

        # The spectrum must cover the entire crop window
        if not covers_halpha_window(wl_min, wl_max):
            return False

        # Filter echelle orders if requested
        if not include_echelle and row.get("is_echelle_order", False):
            return False

        # Filter by SNR if specified
        if min_snr is not None:
            snr = row.get("snr")
            if snr is None or snr < min_snr:
                return False

        return True

    return hf_dataset.filter(keep, num_proc=4)


def split_by_star(hf_dataset, test_stars: list):
    """
    Split the dataset into train/test by star.
    Identical to the full encoder for comparability.
    """
    test_stars_set = set(s.upper() for s in test_stars)

    def is_test(row):
        name = (row.get("star_name") or "").upper()
        return name in test_stars_set

    def is_train(row):
        name = (row.get("star_name") or "").upper()
        return name not in test_stars_set

    train_ds = hf_dataset.filter(is_train, num_proc=4)
    test_ds = hf_dataset.filter(is_test, num_proc=4)
    return train_ds, test_ds


# ── Full pipeline ────────────────────────────────────────────────────────────

def prepare_data(cfg: TrainConfig):
    """
    Full pipeline: HuggingFace → PyTorch DataLoaders.

    Steps:
      1. Load the full dataset from the HuggingFace Hub
      2. Filter spectra covering Hα ±50 Å
      3. Split train/test by star
      4. Create HalphaSpectralDataset instances
      5. Wrap in DataLoaders

    Returns:
      (train_loader, test_loader)
    """
    # Step 1: Load from HF
    print(f"Loading dataset: {cfg.dataset_name}")
    raw = load_dataset(cfg.dataset_name, split="train")
    print(f"  Total spectra: {len(raw):,}")

    # Step 2: Filter Hα spectra
    print(f"Filtering for Hα coverage (±{cfg.halpha_half_window} Å)...")
    print(f"  Include echelle: {cfg.include_echelle}")
    filtered = filter_halpha_spectra(
        raw,
        include_echelle=cfg.include_echelle,
        min_snr=cfg.min_snr,
    )
    print(f"  After filter: {len(filtered):,}")

    # Stats rapides
    n_echelle = sum(1 for r in filtered if r.get("is_echelle_order", False))
    n_single = len(filtered) - n_echelle
    print(f"  Non-echelle: {n_single:,}  |  Echelle orders: {n_echelle:,}")

    # Step 3: Split by star
    print(f"Splitting by star (test stars: {cfg.test_stars})")
    train_hf, test_hf = split_by_star(filtered, cfg.test_stars)
    print(f"  Train: {len(train_hf):,}  |  Test: {len(test_hf):,}")

    # Star stats
    train_stars = set(r["star_name"] for r in train_hf if r.get("star_name"))
    test_stars_actual = set(r["star_name"] for r in test_hf if r.get("star_name"))
    print(f"  Train stars: {len(train_stars):,}  |  Test stars: {len(test_stars_actual):,}")

    # Step 4: Create PyTorch datasets
    train_ds = HalphaSpectralDataset(
        train_hf, n_bins=cfg.n_bins,
        norm_mode=cfg.norm_mode, n_edge_pixels=cfg.n_edge_pixels,
    )
    test_ds = HalphaSpectralDataset(
        test_hf, n_bins=cfg.n_bins,
        norm_mode=cfg.norm_mode, n_edge_pixels=cfg.n_edge_pixels,
    )

    # Step 5: DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, test_loader

#!/usr/bin/env python3
"""
03_push_to_hub.py — Publication of the Dataset on the HuggingFace Hub (private)

Loads the DatasetDict built by 02_build_dataset.py and pushes it
to a HuggingFace Hub repo in **private** mode.

Prerequisites:
  - HuggingFace token with write permissions (role "write")
  - The token may be provided via:
    1. HF_TOKEN environment variable
    2. A .env file at the project root
    3. The --token command-line argument
    4. Prior `huggingface-cli login`

Usage:
    # Via environment variable (recommended)
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
    python 03_push_to_hub.py --repo-id "username/bess-foundation-v1"

    # Via argument
    python 03_push_to_hub.py --repo-id "username/bess-foundation-v1" \\
                             --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"

    # Dry-run (check everything without pushing)
    python 03_push_to_hub.py --repo-id "username/bess-foundation-v1" --dry-run
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi, login

# Load .env if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# ─── Default paths ──────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = Path(os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset"))

# ─── Dataset Card (README.md of the repo) ────────────────────────────────────────
DATASET_CARD = """---
language:
- en
license: cc-by-4.0
task_categories:
- other
tags:
- astronomy
- spectroscopy
- be-stars
- stellar-spectra
- time-series
size_categories:
- 100K<n<1M
---

# BeSS Foundation Dataset — Be Star Spectra

> **339k spectral observations** of Be stars from the BeSS (Be Star Spectra) database,
> cleaned and validated for machine learning research.

## Dataset Description

This dataset contains spectral observations of **Be stars** (B-type stars showing
emission lines, typically Hα) collected by the international BeSS community over
35 years (1990–2025). Each entry contains the full spectral arrays (wavelength, flux,
optional error) alongside cleaned metadata.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total spectra | ~339,000 |
| Unique stars | 1,468 |
| Temporal coverage | 1990–2025 |
| Wavelength range | 1,150–10,442 Å |
| Median SNR | 189 |

### ⚠️ Important: Echelle Orders

**81.6% of entries are individual echelle orders**, not independent observations.
BeSS stores each order of an echelle spectrograph as a separate spectrum entry.
For example, one Elodie observation produces 67 entries.

Use the `is_echelle_order` column to filter:
- `is_echelle_order = False` → 62,476 single-order spectra (independent observations)
- `is_echelle_order = True` → 276,639 individual echelle orders

### Splits

Split by **star** (not by spectrum) to prevent temporal data leakage:
- **train**: ~70% of stars
- **validation**: ~15% of stars
- **test**: ~15% of stars

## Schema

### Spectral Arrays
| Column | Type | Description |
|--------|------|-------------|
| `wavelength` | list[float32] | Native wavelength grid (Å) |
| `flux` | list[float32] | Calibrated flux |
| `flux_error` | list[float32] | Flux error (nullable) |

### Identity
| Column | Type | Description |
|--------|------|-------------|
| `spectrum_id` | string | Unique ID (MD5 hash) |
| `star_name` | string | Canonical star name (deduplicated) |
| `star_name_raw` | string | Original BeSS name |
| `ra`, `dec` | float64 | J2000 coordinates (degrees) |

### Spectral Metadata
| Column | Type | Description |
|--------|------|-------------|
| `lambda_min`, `lambda_max` | float64 | Spectral coverage (Å) |
| `n_pixels` | int32 | Number of spectral pixels |
| `spectral_resolution` | float64 | Theoretical R |
| `spectral_resolution_measured` | float64 | Measured R |

### Quality
| Column | Type | Description |
|--------|------|-------------|
| `snr` | float64 | SNR (DER_SNR method) |
| `snr_continuum` | float64 | SNR from continuum windows |
| `snr_quality` | string | excellent/good/medium/low/very_low/unknown |

### Temporal
| Column | Type | Description |
|--------|------|-------------|
| `observation_date` | string | ISO-8601 date |
| `mjd` | float64 | Modified Julian Date |
| `exposure_time` | float64 | Exposure time (seconds) |

### Instrument
| Column | Type | Description |
|--------|------|-------------|
| `spectrograph` | string | Cleaned spectrograph name |
| `telescope` | string | Cleaned telescope name |
| `observer_type` | string | amateur / professional / unknown |
| `is_echelle_order` | bool | True if individual echelle order |

### Geography & Corrections
| Column | Type | Description |
|--------|------|-------------|
| `site_latitude`, `site_longitude` | float64 | Observer coordinates |
| `helio_velocity` | float64 | Heliocentric correction (km/s) |

## Data Pipeline

Cleaning pipeline applied (in order):
1. **SNR estimation** — DER_SNR (Stoehr et al. 2008) + continuum windows
2. **Instrument cleanup** — Regex parsing → normalized names (99.2% high confidence)
3. **Coordinate validation** — Longitude normalization, elevation checks
4. **MJD validation** — Cross-validation with observation_date
5. **Observer classification** — Professional vs amateur based on spectrograph
6. **Star name normalization** — Textual + spatial cross-ID (4,755 → 1,468 unique)
7. **Echelle order flagging** — Spectrograph-based + structural validation

## Source

[BeSS Database](http://basebe.obspm.fr) — maintained by LESIA, Observatoire de Paris.

## Citation

If you use this dataset, please cite the BeSS database:
> Neiner, C. et al. (2011), "The BeSS database", AJ, 142, 149

## License

CC-BY-4.0 — Please credit the BeSS consortium and individual observers.
"""


def get_token(args_token: str = None) -> str:
    """
    Resolves the HuggingFace token by priority order:
    1. --token argument
    2. HF_TOKEN environment variable
    3. Existing login (huggingface-cli login)
    """
    # 1. CLI argument
    if args_token:
        return args_token

    # 2. Environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # 3. Existing login via huggingface_hub
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Push Dataset to HuggingFace Hub (private)")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HF repo ID (e.g. 'username/bess-foundation-v1')")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                        help="Folder containing the DatasetDict")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or via HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Create the repo as private (default: True)")
    parser.add_argument("--public", action="store_true", default=False,
                        help="Create the repo as public")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check everything without pushing")
    parser.add_argument("--max-shard-size", type=str, default="500MB",
                        help="Max shard size (default: 500MB)")
    args = parser.parse_args()

    # If --public, disable --private
    if args.public:
        args.private = False

    print("=" * 70)
    print("PUSH DATASET → HUGGINGFACE HUB")
    print("=" * 70)
    print(f"  Repo ID  : {args.repo_id}")
    print(f"  Private  : {args.private}")
    print(f"  Source   : {args.dataset_dir}")
    print(f"  Dry-run  : {args.dry_run}")
    print(f"  Date     : {datetime.now().isoformat()}")

    # ── 1. Resolve the token ──
    token = get_token(args.token)
    if not token:
        print("\n  ✗ No HuggingFace token found.")
        print("    Provide one via:")
        print("      1. export HF_TOKEN='hf_...'")
        print("      2. --token 'hf_...'")
        print("      3. huggingface-cli login")
        print("      4. .env file with HF_TOKEN=hf_...")
        sys.exit(1)

    token_preview = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
    print(f"  Token    : {token_preview}")

    # ── 2. Load the dataset ──
    print(f"\n--- Loading DatasetDict ---")
    if not args.dataset_dir.exists():
        print(f"  ✗ Folder not found: {args.dataset_dir}")
        print(f"    Run 02_build_dataset.py first")
        sys.exit(1)

    dataset = load_from_disk(str(args.dataset_dir))
    print(f"  Splits : {', '.join(f'{k}: {len(v):,}' for k, v in dataset.items())}")
    total = sum(len(v) for v in dataset.values())
    print(f"  Total  : {total:,} spectra")

    # ── 3. Quick verification ──
    print(f"\n--- Verification ---")
    for split_name, ds in dataset.items():
        sample = ds[0]
        has_wave = sample.get("wavelength") is not None and len(sample["wavelength"]) > 0
        has_flux = sample.get("flux") is not None and len(sample["flux"]) > 0
        has_name = sample.get("star_name") is not None
        print(f"  {split_name}: wavelength={'✓' if has_wave else '✗'} "
              f"flux={'✓' if has_flux else '✗'} "
              f"star_name={'✓' if has_name else '✗'} "
              f"n_cols={len(sample)}")

    if args.dry_run:
        print(f"\n  [DRY-RUN] Verification successful. No push performed.")
        print(f"  To push, rerun without --dry-run")
        return

    # ── 4. Login ──
    print(f"\n--- HuggingFace authentication ---")
    login(token=token)
    print(f"  ✓ Authenticated")

    # ── 5. Push to Hub ──
    print(f"\n--- Pushing to {args.repo_id} ---")
    t0 = time.time()

    dataset.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        token=token,
        max_shard_size=args.max_shard_size,
    )

    # ── 6. Upload the dataset card ──
    print(f"\n--- Uploading Dataset Card ---")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )

    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"PUSH DONE")
    print(f"{'=' * 70}")
    print(f"  Repo     : https://huggingface.co/datasets/{args.repo_id}")
    print(f"  Private  : {args.private}")
    print(f"  Duration : {elapsed / 60:.1f} min")
    print(f"  Spectra  : {total:,}")
    print(f"\n  ✓ Dataset available on HuggingFace Hub")


if __name__ == "__main__":
    main()

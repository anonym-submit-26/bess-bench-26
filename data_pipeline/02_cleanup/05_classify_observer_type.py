#!/usr/bin/env python3
"""
05_classify_observer_type.py — Professional / amateur classification

Adds an `observer_type` column based on the spectrograph used.

Classification criteria:
- The **spectrograph** is the most reliable indicator because:
  1. It determines the instrumental quality (stability, calibration, resolution)
  2. It is independent of the site (an eShel at OHP remains an amateur eShel)
  3. Professional spectrographs are documented in the literature
  4. They are tied to institutional observation programs with validated
     reduction pipelines

Categories:
- "professional"  : institutional spectrograph on a dedicated telescope
                    (validated reduction pipeline, publication-grade calibration)
- "amateur"       : commercial or homemade spectrograph
- "unknown"       : impossible to determine (spectrograph_clean missing)

Note: this is NOT a quality judgment. An expert amateur with a well-calibrated
eShel can produce better-quality spectra than an old IUE spectrum.
This is a classification by **instrumental origin**, useful to:
- Filter/stratify the dataset
- Study systematic biases tied to instrumentation
- Build homogeneous subsets for training

Usage:
    python 05_classify_observer_type.py [--dry-run] [--db PATH]
"""

import argparse
import sys
from pathlib import Path

import duckdb

DEFAULT_DB = Path(__file__).parent.parent.parent / "harvest" / "bess_data.duckdb"

# ─── Known professional spectrographs ─────────────────────────────────────────
# Each entry is documented with the associated telescope and reference.
#
# Inclusion criteria:
#   1. Instrument built for/by a professional observatory
#   2. Mounted on a telescope of class ≥1m (or satellite)
#   3. Official reduction pipeline (DRS, MIDAS, etc.)
#   4. Peer-reviewed scientific publications using the instrument
#
PROFESSIONAL_SPECTROGRAPHS = {
    # ── OHP (Haute-Provence Observatory) ──
    "Elodie": {
        "telescope": "OHP T193 (1.93m)",
        "type": "fiber-fed echelle",
        "R": 42_000,
        "reference": "Baranne et al. 1996, A&AS 119, 373",
        "note": "Predecessor of SOPHIE, in service 1993-2006",
    },
    "SOPHIE": {
        "telescope": "OHP T193 (1.93m)",
        "type": "fiber-fed echelle",
        "R": 75_000,
        "reference": "Perruchot et al. 2008, SPIE 7014",
        "note": "Successor of ELODIE, in service since 2006",
    },
    "Aurelie": {
        "telescope": "OHP T152 (1.52m)",
        "type": "grating spectrograph",
        "R": "5000-30000",
        "reference": "Gillet et al. 1994, A&AS 108, 181",
    },

    # ── Pic du Midi / TBL ──
    "MUSICOS": {
        "telescope": "TBL 2m (Pic du Midi)",
        "type": "fiber-fed echelle spectropolarimeter",
        "R": 35_000,
        "reference": "Baudrand & Bohm 1992, A&A 259, 711",
        "note": "In service 1990-2006 (then NARVAL)",
    },
    "NARVAL": {
        "telescope": "TBL 2m (Pic du Midi)",
        "type": "echelle spectropolarimeter",
        "R": 65_000,
        "reference": "Auriere 2003, EAS Pub. Series 9, 105",
        "note": "Successor of MUSICOS, in service since 2006",
    },

    # ── ESO (La Silla, Paranal) ──
    "FEROS": {
        "telescope": "ESO 1.52m / MPG 2.2m (La Silla)",
        "type": "fiber-fed echelle",
        "R": 48_000,
        "reference": "Kaufer et al. 1999, The Messenger 95, 8",
    },
    "HARPS": {
        "telescope": "ESO 3.6m (La Silla)",
        "type": "vacuum echelle",
        "R": 115_000,
        "reference": "Mayor et al. 2003, The Messenger 114, 20",
    },
    "UVES": {
        "telescope": "VLT UT2 8.2m (Paranal)",
        "type": "UV-Visible echelle",
        "R": 110_000,
        "reference": "Dekker et al. 2000, SPIE 4008, 534",
    },
    "X-shooter": {
        "telescope": "VLT UT3 8.2m (Paranal)",
        "type": "multi-arm echelle (UVB+VIS+NIR)",
        "R": "5000-18000",
        "reference": "Vernet et al. 2011, A&A 536, A105",
    },

    # ── CFHT (Mauna Kea) ──
    "ESPaDOnS": {
        "telescope": "CFHT 3.6m (Mauna Kea)",
        "type": "echelle spectropolarimeter",
        "R": 68_000,
        "reference": "Donati et al. 2006, ASP Conf. Series 358, 362",
    },

    # ── Italy (TNG, La Palma) ──
    "SARG": {
        "telescope": "TNG 3.58m (La Palma)",
        "type": "high-resolution echelle",
        "R": 46_000,
        "reference": "Gratton et al. 2001, Exp. Astron. 12, 107",
    },

    # ── Satellite ──
    "IUE_SWP": {
        "telescope": "IUE (satellite)",
        "type": "UV echelle (1150-1980 Å)",
        "R": "10000-13000",
        "reference": "Boggess et al. 1978, Nature 275, 372",
        "note": "International Ultraviolet Explorer, 1978-1996",
    },
    "IUE_LWP": {
        "telescope": "IUE (satellite)",
        "type": "UV echelle (1850-3350 Å)",
        "R": "10000-13000",
        "reference": "Boggess et al. 1978, Nature 275, 372",
    },

    # ── Other observatories ──
    "HEROS": {
        "telescope": "Various (Heidelberg, La Silla, Ondrejov)",
        "type": "double echelle (blue+red)",
        "R": 20_000,
        "reference": "Kaufer 1998, PhD thesis, Heidelberg",
        "note": "Transportable spectrograph, used at ESO, Ondrejov, etc.",
    },

    # ── Nordic / Mercator ──
    "FIES": {
        "telescope": "NOT 2.56m (La Palma)",
        "type": "fiber-fed echelle",
        "R": 67_000,
        "reference": "Telting et al. 2014, AN 335, 41",
    },
    "HERMES": {
        "telescope": "Mercator 1.2m (La Palma)",
        "type": "fiber-fed echelle",
        "R": 85_000,
        "reference": "Raskin et al. 2011, A&A 526, A69",
    },

    # ── Spain ──
    "CAFE": {
        "telescope": "Calar Alto 2.2m",
        "type": "fiber-fed echelle",
        "R": 62_000,
        "reference": "Aceituno et al. 2013, A&A 552, A31",
    },

    # ── South America ──
    "BESO": {
        "telescope": "Hexapod 1.5m (Cerro Murphy, Chile)",
        "type": "echelle (former HEROS blue)",
        "R": 48_000,
        "reference": "Stefl et al. 2008, in prep",
    },
}


def classify_spectra(con: duckdb.DuckDBPyConnection, dry_run: bool = False) -> dict:
    """Classify each spectrum as professional/amateur/unknown."""

    # Build the CASE WHEN clause from the dictionary
    # Match on spectrograph_clean (normalized by the instruments pipeline)
    pro_names = list(PROFESSIONAL_SPECTROGRAPHS.keys())

    # Check which pro spectrographs are actually present in the data
    placeholders = ", ".join([f"'{name}'" for name in pro_names])
    found = con.execute(f"""
        SELECT DISTINCT spectrograph_clean, COUNT(*) as n
        FROM spectra_votable
        WHERE spectrograph_clean IN ({placeholders})
        GROUP BY spectrograph_clean
        ORDER BY n DESC
    """).fetchdf()

    print("Professional spectrographs found in the data:")
    for _, row in found.iterrows():
        info = PROFESSIONAL_SPECTROGRAPHS.get(row['spectrograph_clean'], {})
        print(f"  {row['spectrograph_clean']:15s} : {row['n']:>6,} spectra  "
              f"({info.get('telescope', '?')})")

    # Count expected distribution
    dist = con.execute(f"""
        SELECT
            CASE
                WHEN spectrograph_clean IN ({placeholders}) THEN 'professional'
                WHEN spectrograph_clean IS NOT NULL THEN 'amateur'
                ELSE 'unknown'
            END as observer_type,
            COUNT(*) as n
        FROM spectra_votable
        GROUP BY observer_type
        ORDER BY n DESC
    """).fetchdf()

    print("\nExpected distribution:")
    for _, row in dist.iterrows():
        print(f"  {row['observer_type']:15s} : {row['n']:>7,}")

    if dry_run:
        return {
            "status": "dry_run",
            "distribution": dist.to_dict('records'),
            "pro_spectrographs_found": found.to_dict('records'),
        }

    # Add the column
    con.execute("""
        ALTER TABLE spectra_votable
        ADD COLUMN IF NOT EXISTS observer_type VARCHAR
    """)

    # Apply the classification
    con.execute(f"""
        UPDATE spectra_votable
        SET observer_type = CASE
            WHEN spectrograph_clean IN ({placeholders}) THEN 'professional'
            WHEN spectrograph_clean IS NOT NULL THEN 'amateur'
            ELSE 'unknown'
        END
    """)

    # Verification
    final_dist = con.execute("""
        SELECT observer_type, COUNT(*) as n
        FROM spectra_votable
        GROUP BY observer_type
        ORDER BY n DESC
    """).fetchdf()

    print("\n✓ Column observer_type added")
    print("\nFinal distribution:")
    for _, row in final_dist.iterrows():
        pct = 100 * row['n'] / final_dist['n'].sum()
        print(f"  {row['observer_type']:15s} : {row['n']:>7,} ({pct:.1f}%)")

    return {
        "status": "applied",
        "distribution": final_dist.to_dict('records'),
        "pro_spectrographs_found": found.to_dict('records'),
    }


def validate_classification(con: duckdb.DuckDBPyConnection):
    """A few cross-checks to verify consistency."""

    print("\n=== Cross-validation ===")

    # 1. Pro vs amateur median SNR
    print("\n1. Median SNR by category:")
    snr = con.execute("""
        SELECT
            observer_type,
            COUNT(*) as n,
            ROUND(MEDIAN(snr_estimated), 1) as median_snr,
            ROUND(AVG(snr_estimated), 1) as mean_snr
        FROM spectra_votable
        WHERE snr_estimated IS NOT NULL
        GROUP BY observer_type
        ORDER BY median_snr DESC
    """).fetchdf()
    print(snr.to_string(index=False))

    # 2. Spectral resolution by category
    print("\n2. Median spectral resolution by category:")
    res = con.execute("""
        SELECT
            observer_type,
            ROUND(MEDIAN(fits_bss_itrp), 0) as median_R,
            ROUND(AVG(fits_bss_itrp), 0) as mean_R
        FROM spectra_votable
        WHERE fits_bss_itrp IS NOT NULL
        GROUP BY observer_type
        ORDER BY median_R DESC
    """).fetchdf()
    print(res.to_string(index=False))

    # 3. The "unknown" — what is it?
    print("\n3. 'unknown' spectra — source instruments:")
    unknown = con.execute("""
        SELECT fits_bss_inst, COUNT(*) as n
        FROM spectra_votable
        WHERE observer_type = 'unknown'
        GROUP BY fits_bss_inst
        ORDER BY n DESC
        LIMIT 10
    """).fetchdf()
    print(unknown.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Professional/amateur classification of spectra")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without modification")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to the DuckDB database")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"✗ Database not found: {args.db}")
        sys.exit(1)

    print(f"Database: {args.db}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print()

    con = duckdb.connect(str(args.db), read_only=args.dry_run)

    result = classify_spectra(con, dry_run=args.dry_run)

    if not args.dry_run:
        validate_classification(con)

    print("\n✓ Done")


if __name__ == "__main__":
    main()

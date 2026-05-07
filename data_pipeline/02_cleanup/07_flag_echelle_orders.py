#!/usr/bin/env python3
"""
07_flag_echelle_orders.py — Identification of echelle spectrograph orders

Adds a column:
  - `is_echelle_order` (BOOLEAN): True if the spectrum is an individual order
    of an echelle spectrograph, False otherwise.

Context:
  BeSS stores each order of an echelle spectrograph as an individual spectrum.
  One observation with Elodie (67 orders) generates 67 entries in the database.
  This represents ~81% of the 339k spectra.

  This information is crucial for:
    1. HuggingFace export: documenting that the majority of entries are orders
    2. Filtering: allowing work on single-order spectra only
    3. Statistics: not confusing "number of spectra" with "number of observations"

Detection method (2 criteria combined):
  1. Spectrograph known to be echelle (manually maintained list)
     → Elodie, MUSICOS, VHIRES, eShel, ESP, FEROS, DeniSe, Dubs_echelle, etc.
  2. Structural validation: the spectrum belongs to a group of spectra
     sharing the same (star_name_clean, MJD, num_points) but with different
     λ_start values.
     → Confirms that the spectrograph indeed produces multiple orders per
     observation.

  Criterion 1 is sufficient in practice (99%+ coverage); criterion 2 acts as
  validation and can detect echelle spectrographs that are not yet listed.

Usage:
    python 07_flag_echelle_orders.py [--dry-run] [--db PATH]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).parent.parent.parent / "harvest" / "bess_data.duckdb"

# ─── Known echelle spectrographs in BeSS ────────────────────────────────────
# List established by the pre-export audit (analysis of strict-duplicate
# groups with Δλ_start > 10Å between orders of the same group).
#
# Each entry: (clean_spectrograph_name, typical_n_orders)
ECHELLE_SPECTROGRAPHS = {
    "Elodie":        67,   # OHP 1.93m, 67 orders, R~42000
    "MUSICOS":       40,   # Multi-Site, ~40 orders, R~35000
    "VHIRES":        18,   # VHIRES, ~18 orders
    "eShel":          2,   # Shelyak eShel, ~24 orders (often exported as 2 segments)
    "ESP":           15,   # ESPaDOnS / CFHT, ~40 orders (variable in BeSS)
    "FEROS":         39,   # ESO/MPG 2.2m, 39 orders, R~48000
    "DeniSe":         3,   # DeniSe instrument
    "Dubs_echelle":   4,   # Dubs echelle spectrograph
}


def flag_echelle_orders(con: duckdb.DuckDBPyConnection, dry_run: bool = False) -> dict:
    """
    Identifies and flags echelle spectrograph orders.

    Returns a dict with the operation statistics.
    """
    print("=" * 70)
    print("MODULE 07 — FLAG ECHELLE ORDERS")
    print("=" * 70)

    # ── Check whether the column already exists ──
    cols = [row[0] for row in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'spectra_votable'"
    ).fetchall()]

    if "is_echelle_order" in cols:
        n_true = con.execute(
            "SELECT COUNT(*) FROM spectra_votable WHERE is_echelle_order = true"
        ).fetchone()[0]
        n_false = con.execute(
            "SELECT COUNT(*) FROM spectra_votable WHERE is_echelle_order = false"
        ).fetchone()[0]
        print(f"\n  ⚠ Column is_echelle_order already exists")
        print(f"    True  (echelle) : {n_true:,}")
        print(f"    False (single)  : {n_false:,}")
        print(f"    → Recomputing for update...")

    # ── Load the required data ──
    print(f"\n--- Loading data ---")
    df = con.execute("""
        SELECT id, spectrograph_clean, star_name_clean, 
               time_location_mjd, num_points, 
               spectral_start, spectral_stop
        FROM spectra_votable
    """).fetchdf()
    n_total = len(df)
    print(f"  Total spectra: {n_total:,}")

    # ── Criterion 1: known echelle spectrograph ──
    print(f"\n--- Criterion 1: known echelle spectrographs ---")
    echelle_names = set(ECHELLE_SPECTROGRAPHS.keys())
    df["is_echelle_spectro"] = df["spectrograph_clean"].isin(echelle_names)

    n_by_spectro = df[df["is_echelle_spectro"]].groupby("spectrograph_clean").size()
    n_echelle_c1 = df["is_echelle_spectro"].sum()

    print(f"  Listed echelle spectrographs : {len(echelle_names)}")
    print(f"  Spectra matched (criterion 1): {n_echelle_c1:,} ({100*n_echelle_c1/n_total:.1f}%)")
    print(f"\n  Per-spectrograph detail:")
    for spectro, count in n_by_spectro.sort_values(ascending=False).items():
        expected = ECHELLE_SPECTROGRAPHS.get(spectro, "?")
        print(f"    {spectro:<20} : {count:>7,} spectra  "
              f"(~{expected} orders/obs → ~{count//expected:,} observations)")

    # ── Criterion 2: structural validation ──
    # Groups (star, MJD, npts) with > 1 member AND Δλ_start > 10Å
    print(f"\n--- Criterion 2: structural validation ---")
    M_TO_A = 1e10

    dup_key = ["star_name_clean", "time_location_mjd", "num_points"]
    dup_mask = df.duplicated(subset=dup_key, keep=False)
    dup_df = df[dup_mask].copy()

    # For each group, check whether λ_start varies (= distinct orders)
    groups = dup_df.groupby(dup_key)
    echelle_ids_structural = set()

    n_groups_echelle = 0
    n_groups_true_dup = 0
    unexpected_spectros = set()

    for (star, mjd, npts), group in groups:
        wl_starts = group["spectral_start"].values * M_TO_A
        wl_range = np.ptp(wl_starts) if len(wl_starts) > 1 else 0

        if wl_range > 10:  # Δλ > 10Å → distinct orders
            n_groups_echelle += 1
            echelle_ids_structural.update(group["id"].values)

            # Check whether the spectrograph is in our list
            spectros = group["spectrograph_clean"].unique()
            for s in spectros:
                if s not in echelle_names:
                    unexpected_spectros.add(s)
        else:
            n_groups_true_dup += 1

    n_echelle_c2 = len(echelle_ids_structural)
    print(f"  Structural echelle groups    : {n_groups_echelle:,}")
    print(f"  True-duplicate groups        : {n_groups_true_dup:,}")
    print(f"  Spectra matched (criterion 2): {n_echelle_c2:,}")

    if unexpected_spectros:
        print(f"\n  ⚠ Spectrographs NOT listed but detected as echelle:")
        for s in sorted(unexpected_spectros):
            n_s = df[(df["spectrograph_clean"] == s) &
                     (df["id"].isin(echelle_ids_structural))].shape[0]
            print(f"    {s:<25} : {n_s:,} spectra")
        print(f"    → Consider adding to ECHELLE_SPECTROGRAPHS")

    # ── Combine the two criteria ──
    # A spectrum is echelle if criterion 1 OR criterion 2
    print(f"\n--- Combining criteria ---")
    df["is_echelle_c2"] = df["id"].isin(echelle_ids_structural)
    df["is_echelle_order"] = df["is_echelle_spectro"] | df["is_echelle_c2"]

    n_final = df["is_echelle_order"].sum()
    n_c1_only = (df["is_echelle_spectro"] & ~df["is_echelle_c2"]).sum()
    n_c2_only = (~df["is_echelle_spectro"] & df["is_echelle_c2"]).sum()
    n_both = (df["is_echelle_spectro"] & df["is_echelle_c2"]).sum()

    print(f"  Criterion 1 only : {n_c1_only:,}")
    print(f"  Criterion 2 only : {n_c2_only:,}")
    print(f"  Both             : {n_both:,}")
    print(f"  TOTAL echelle    : {n_final:,} ({100*n_final/n_total:.1f}%)")
    print(f"  Single-order     : {n_total - n_final:,} ({100*(n_total-n_final)/n_total:.1f}%)")

    # ── Apply to DuckDB ──
    if dry_run:
        print(f"\n  [DRY-RUN] No modification applied")
        return {
            "status": "dry-run",
            "n_total": n_total,
            "n_echelle": int(n_final),
            "n_single_order": int(n_total - n_final),
            "unexpected_spectros": list(unexpected_spectros),
        }

    print(f"\n--- Applying to DuckDB ---")

    # Create the column if necessary
    if "is_echelle_order" not in cols:
        con.execute("ALTER TABLE spectra_votable ADD COLUMN is_echelle_order BOOLEAN")
        print(f"  ✓ Column is_echelle_order created")
    else:
        print(f"  ✓ Column is_echelle_order exists, updating")

    # Initialise everything to FALSE
    con.execute("UPDATE spectra_votable SET is_echelle_order = false")

    # Mark echelle as TRUE
    echelle_ids = df.loc[df["is_echelle_order"], "id"].tolist()

    # In batches to avoid overly long queries
    batch_size = 10000
    n_updated = 0
    for i in range(0, len(echelle_ids), batch_size):
        batch = echelle_ids[i : i + batch_size]
        placeholders = ",".join([f"'{x}'" for x in batch])
        con.execute(f"""
            UPDATE spectra_votable
            SET is_echelle_order = true
            WHERE id IN ({placeholders})
        """)
        n_updated += len(batch)
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size + 1}: {n_updated:,} / {len(echelle_ids):,}")

    print(f"  ✓ {n_updated:,} spectra marked is_echelle_order = true")

    # ── Verification ──
    print(f"\n--- Verification ---")
    check = con.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN is_echelle_order = true THEN 1 ELSE 0 END) as n_echelle,
            SUM(CASE WHEN is_echelle_order = false THEN 1 ELSE 0 END) as n_single
        FROM spectra_votable
    """).fetchone()

    print(f"  Total          : {check[0]:,}")
    print(f"  Echelle (true) : {check[1]:,} ({100*check[1]/check[0]:.1f}%)")
    print(f"  Single (false) : {check[2]:,} ({100*check[2]/check[0]:.1f}%)")

    # Cross-tab with observer_type
    cross = con.execute("""
        SELECT observer_type, is_echelle_order,
               COUNT(*) as n
        FROM spectra_votable
        GROUP BY observer_type, is_echelle_order
        ORDER BY observer_type, is_echelle_order
    """).fetchdf()
    print(f"\n  Cross-tab observer_type × is_echelle_order:")
    for _, row in cross.iterrows():
        label = "echelle" if row["is_echelle_order"] else "single "
        print(f"    {str(row['observer_type']):15s}  {label} : {row['n']:>8,}")

    # Per-spectrograph stats
    stats = con.execute("""
        SELECT spectrograph_clean, 
               SUM(CASE WHEN is_echelle_order THEN 1 ELSE 0 END) as n_echelle,
               SUM(CASE WHEN NOT is_echelle_order THEN 1 ELSE 0 END) as n_single,
               COUNT(*) as total
        FROM spectra_votable
        GROUP BY spectrograph_clean
        HAVING SUM(CASE WHEN is_echelle_order THEN 1 ELSE 0 END) > 0
        ORDER BY n_echelle DESC
    """).fetchdf()
    print(f"\n  Spectrographs with echelle orders:")
    for _, row in stats.iterrows():
        pct = 100 * row["n_echelle"] / row["total"]
        print(f"    {row['spectrograph_clean']:<25} : {row['n_echelle']:>7,} echelle / "
              f"{row['total']:>7,} total ({pct:.0f}%)")

    return {
        "status": "applied",
        "n_total": int(check[0]),
        "n_echelle": int(check[1]),
        "n_single_order": int(check[2]),
        "unexpected_spectros": list(unexpected_spectros),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Flag echelle spectrograph orders in BeSS")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulation with no modification")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to the DuckDB database")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"✗ Database not found: {args.db}")
        sys.exit(1)

    print(f"Database: {args.db}")
    print(f"Mode    : {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print(f"Date    : {datetime.now().isoformat()}")
    print()

    con = duckdb.connect(str(args.db), read_only=args.dry_run)

    result = flag_echelle_orders(con, dry_run=args.dry_run)

    con.close()

    print(f"\n{'='*70}")
    if result["status"] == "applied":
        print(f"✓ Done — {result['n_echelle']:,} echelle orders flagged "
              f"({result['n_single_order']:,} single-order)")
    else:
        print(f"✓ Dry-run — {result['n_echelle']:,} echelle orders detected "
              f"({result['n_single_order']:,} single-order)")
        print("  Rerun without --dry-run to apply.")

    if result["unexpected_spectros"]:
        print(f"\n  ⚠ Unlisted echelle spectrographs detected:")
        for s in result["unexpected_spectros"]:
            print(f"    → {s}")
        print("    Consider adding them to ECHELLE_SPECTROGRAPHS in the script.")


if __name__ == "__main__":
    main()

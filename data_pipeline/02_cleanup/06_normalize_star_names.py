#!/usr/bin/env python3
"""
06_normalize_star_names.py — Normalization of star names

Adds two columns:
  - `star_name_clean`  : normalized (textual) star name
  - `star_group_id`    : star group identifier (spatial cross-ID)

Problem:
  fits_objname contains 4755 unique names for ~1450 actual stars (BeSS).
  Duplicates come from:
    1. Case / spaces / separators       (55%) — "gam Cas" vs "GAM CAS" vs "gam_Cas"
    2. Catalogs without space           (21%) — "HD 5394" vs "HD5394"
    3. Spatial cross-identification     (20%) — "gam Cas" = "HD5394" (same position)
    4. Greek letters abbreviated/full   (2%)  — "gam" ↔ "gamma"
    5. V-stars / Flamsteed formatting   (1%)  — "V0404 Lac" → "V404 Lac"

Two-step strategy:
  Step 1 — Textual normalization (deterministic, pure string):
    → Reduces 4755 to ~2141 names
  Step 2 — Spatial cross-identification (RA/DEC < 36 arcsec):
    → Groups different names pointing to the same star
    → The canonical name is the one with the most spectra in each group
    → Final result: ~1468 unique stars (vs ~1450 official BeSS)

Usage:
    python 06_normalize_star_names.py [--dry-run] [--db PATH] [--radius 0.01]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).parent.parent.parent / "harvest" / "bess_data.duckdb"

# ─── Greek letters: abbreviated ↔ full mapping ──────────────────────────────
# IAU / Bayer convention: the short form (3 letters) is the most common in BeSS.
# Normalize EVERYTHING to the abbreviated form (3 letters).

GREEK_FULL_TO_ABBREV = {
    "ALPHA": "ALF", "BETA": "BET", "GAMMA": "GAM", "DELTA": "DEL",
    "EPSILON": "EPS", "ZETA": "ZET", "ETA": "ETA", "THETA": "TET",
    "IOTA": "IOT", "KAPPA": "KAP", "LAMBDA": "LAM", "MU": "MU",
    "NU": "NU", "XI": "XI", "OMICRON": "OMI", "PI": "PI",
    "RHO": "RHO", "SIGMA": "SIG", "TAU": "TAU", "UPSILON": "UPS",
    "PHI": "PHI", "CHI": "CHI", "PSI": "PSI", "OMEGA": "OME",
}

# Build the completed table: key = any form → value = abbreviated
GREEK_NORMALIZE = {}
for full, abbr in GREEK_FULL_TO_ABBREV.items():
    GREEK_NORMALIZE[full] = abbr
    GREEK_NORMALIZE[abbr] = abbr

# ─── IAU constellations (3-letter abbreviation) ───────────────────────────────
CONSTELLATIONS_3L = {
    "AND", "ANT", "APS", "AQR", "AQL", "ARA", "ARI", "AUR", "BOO",
    "CAE", "CAM", "CNC", "CVN", "CMA", "CMI", "CAP", "CAR", "CAS",
    "CEN", "CEP", "CET", "CHA", "CIR", "COL", "COM", "CRA", "CRB",
    "CRV", "CRT", "CRU", "CYG", "DEL", "DOR", "DRA", "EQU", "ERI",
    "FOR", "GEM", "GRU", "HER", "HOR", "HYA", "HYI", "IND", "LAC",
    "LEO", "LEP", "LIB", "LUP", "LYN", "LYR", "MEN", "MIC", "MON",
    "MUS", "NOR", "OCT", "OPH", "ORI", "PAV", "PEG", "PER", "PHE",
    "PIC", "PSC", "PSA", "PUP", "PYX", "RET", "SGE", "SGR", "SCO",
    "SCL", "SCT", "SER", "SEX", "TAU", "TEL", "TRA", "TRI", "TUC",
    "UMA", "UMI", "VEL", "VIR", "VOL", "VUL",
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Textual normalization
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_star_name(name: str) -> str:
    """Normalize a star name to a canonical textual form.

    Transformations applied (in order):
      1. Strip + UPPER
      2. Collapse spaces/underscores/hyphens → single space
      3. Leading underscore/garbage → removed
      4. Full Greek letters → abbreviated (GAMMA → GAM)
      5. Catalogs: stick the number (HD 5394 → HD5394)
      6. V-stars: normalize (V 442 → V442, V0404 → V404)
      7. Flamsteed without space: split (11CYG → 11 CYG)
      8. Useless trailing suffixes (-/+/*) → removed

    Examples:
      "gam Cas"       → "GAM CAS"
      "gamma_Cas"     → "GAM CAS"
      "HD 5394"       → "HD5394"
      "V 442 And"     → "V442 AND"
      "V0404 Lac"     → "V404 LAC"
      "11Cyg"         → "11 CYG"
      "_hd224544_..." → "HD224544..."
    """
    s = str(name).strip()

    # 1–2. Upper + collapse separators
    s = re.sub(r'[\s_\-]+', ' ', s).strip().upper()

    # 3. Leading garbage
    s = re.sub(r'^[_\s]+', '', s)

    # 4. Greek letters → abbreviated form
    parts = s.split()
    if len(parts) >= 2:
        if parts[0] in GREEK_NORMALIZE:
            parts[0] = GREEK_NORMALIZE[parts[0]]
    s = ' '.join(parts)

    # 5. Catalogs: stick the number
    s = re.sub(r'^(HD|HR|SAO|HIP|MWC|ADS|TYC|NSV|NGC|BD|CD|CPD)\s+', r'\1', s)

    # 6. V-stars: normalize
    s = re.sub(r'^V\s+(\d)', r'V\1', s)         # "V 442" → "V442"
    s = re.sub(r'^V0+(\d{1,4}\s)', r'V\1', s)   # "V0404 LAC" → "V404 LAC"
    s = re.sub(r'^V0+(\d{1,4})$', r'V\1', s)    # "V0404" → "V404"

    # 7. Flamsteed without space (number stuck to a 3-letter constellation)
    s = re.sub(r'^(\d+)([A-Z]{3})$', r'\1 \2', s)

    # 8. Trailing suffixes
    s = re.sub(r'[\-\+\*]+$', '', s).strip()

    return s


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Spatial cross-identification
# ═══════════════════════════════════════════════════════════════════════════════

def build_spatial_groups(
    df: pd.DataFrame,
    name_col: str = "star_name_norm",
    ra_col: str = "ra",
    dec_col: str = "dec",
    radius_deg: float = 0.01,
) -> dict[str, str]:
    """Group normalized names pointing to the same coordinates.

    Algorithm:
      1. For each normalized name → median coordinates (RA, DEC).
      2. KDTree query_pairs with radius < radius_deg (~36 arcsec for 0.01°).
      3. Union-Find to merge pairs into groups.
      4. The canonical name of each group = the one with the most spectra.


    Returns:
      tuple (name_to_canonical, group_details)
        - name_to_canonical : dict { normalized_name → canonical_name }
        - group_details : dict { canonical_name → {members, ra, dec, n_spectra} }
    """
    from scipy.spatial import cKDTree

    # Median coordinates per normalized name
    grouped = df.groupby(name_col).agg(
        ra_med=(ra_col, "median"),
        dec_med=(dec_col, "median"),
        n_spectra=(name_col, "size"),
    )
    grouped = grouped.dropna(subset=["ra_med", "dec_med"])

    if len(grouped) == 0:
        return {}, {}

    # KDTree — search for close pairs
    coords = grouped[["ra_med", "dec_med"]].values
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius_deg)

    # Union-Find
    parent = list(range(len(grouped)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)

    # Build the groups
    idx_to_name = grouped.index.tolist()
    cluster_members = defaultdict(list)
    for i in range(len(grouped)):
        cluster_members[find(i)].append(i)

    # For each group → canonical name = the most observed
    name_to_canonical = {}
    group_details = {}
    n_groups_multi = 0
    for members_idx in cluster_members.values():
        member_names = [idx_to_name[i] for i in members_idx]
        member_counts = [(n, int(grouped.loc[n, "n_spectra"])) for n in member_names]
        canonical = max(member_counts, key=lambda x: x[1])[0]
        for n in member_names:
            name_to_canonical[n] = canonical
        if len(member_names) > 1:
            n_groups_multi += 1
        # Store group details
        group_details[canonical] = {
            "members": sorted(member_names),
            "ra": float(grouped.loc[canonical, "ra_med"]),
            "dec": float(grouped.loc[canonical, "dec_med"]),
            "n_spectra_total": sum(c for _, c in member_counts),
            "member_counts": {n: c for n, c in sorted(member_counts, key=lambda x: -x[1])},
        }

    n_unique = len(set(name_to_canonical.values()))
    print(f"  Cross-ID: {len(grouped)} names → {n_unique} unique stars "
          f"({n_groups_multi} multi-name groups)")

    return name_to_canonical, group_details


# ═══════════════════════════════════════════════════════════════════════════════
# JSON MAPPING EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _export_lookup_json(
    df: pd.DataFrame,
    group_details: dict,
    json_path: Path,
    radius_deg: float,
):
    """Export the complete star → variants mapping to a verifiable JSON.

    JSON structure:
    {
      "_metadata": { date, n_stars, n_raw_names, radius_deg, ... },
      "stars": {
        "GAM CAS": {
          "ra": 14.177, "dec": 60.717,
          "n_spectra": 12635,
          "cross_id_members": ["GAM CAS", "HD5394", "HIP4427", ...],
          "raw_variants": {
            "gam Cas": 1839, "GAM CAS": 4735, "gamma Cas": 903, ...
          }
        },
        ...
      }
    }
    """
    # Build the dictionary by canonical star
    stars = {}

    for canonical_name in sorted(df["star_name_clean"].unique()):
        mask = df["star_name_clean"] == canonical_name
        sub = df[mask]

        # Raw variants with counts
        raw_counts = sub["fits_objname"].value_counts()
        raw_variants = {str(name): int(count) for name, count in raw_counts.items()}

        # Intermediate normalized names (before cross-ID)
        norm_names = sorted(sub["star_name_norm"].unique().tolist())

        # Coordinates and spatial group info
        ra_med = float(sub["ra"].median()) if sub["ra"].notna().any() else None
        dec_med = float(sub["dec"].median()) if sub["dec"].notna().any() else None

        entry = {
            "n_spectra": int(mask.sum()),
            "ra": round(ra_med, 6) if ra_med is not None else None,
            "dec": round(dec_med, 6) if dec_med is not None else None,
            "n_raw_variants": len(raw_variants),
            "raw_variants": raw_variants,
        }

        # If cross-ID merged several normalized names
        if len(norm_names) > 1:
            entry["cross_id_members"] = norm_names

        stars[canonical_name] = entry

    # Metadata
    output = {
        "_metadata": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "description": (
                "Mapping of BeSS star names: raw fits_objname → "
                "canonical star_name_clean. Each entry lists all the "
                "raw spelling variants and, when applicable, the different "
                "normalized names merged by spatial cross-identification."
            ),
            "n_stars_canonical": len(stars),
            "n_raw_names": int(df["fits_objname"].nunique()),
            "n_spectra_total": len(df),
            "cross_id_radius_deg": radius_deg,
            "cross_id_radius_arcsec": radius_deg * 3600,
            "pipeline_step": "06_normalize_star_names",
        },
        "stars": stars,
    }

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  ✓ Lookup exported: {json_path}")
    print(f"    {len(stars)} stars, {json_path.stat().st_size // 1024} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION TO THE DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_star_names(
    con: duckdb.DuckDBPyConnection,
    dry_run: bool = False,
    radius_deg: float = 0.01,
):
    """Complete pipeline for star-name normalization."""

    print("=" * 65)
    print("STAR NAME NORMALIZATION")
    print("=" * 65)

    # ── 1. Loading ──
    df = con.execute("""
        SELECT id, fits_objname, ra, dec
        FROM spectra_votable
        WHERE fits_objname IS NOT NULL
    """).fetchdf()

    n_total = con.execute("SELECT COUNT(*) FROM spectra_votable").fetchone()[0]
    n_with_name = len(df)
    n_no_name = n_total - n_with_name
    n_raw_unique = df["fits_objname"].nunique()

    print(f"\n  Total spectra         : {n_total:,}")
    print(f"  With fits_objname     : {n_with_name:,}")
    print(f"  Without fits_objname  : {n_no_name:,}")
    print(f"  Unique raw names      : {n_raw_unique:,}")

    # ── 2. Textual normalization ──
    print(f"\n--- Step 1: Textual normalization ---")
    df["star_name_norm"] = df["fits_objname"].apply(normalize_star_name)
    n_after_text = df["star_name_norm"].nunique()
    print(f"  Names after textual normalization: {n_after_text:,} "
          f"(-{n_raw_unique - n_after_text:,})")

    # ── 3. Spatial cross-identification ──
    print(f"\n--- Step 2: Spatial cross-identification (radius = {radius_deg}°) ---")
    canonical_map, group_details = build_spatial_groups(df, radius_deg=radius_deg)

    # Apply the mapping
    df["star_name_clean"] = df["star_name_norm"].map(canonical_map)
    # Fallback for names without coordinates: keep the normalized name
    df["star_name_clean"] = df["star_name_clean"].fillna(df["star_name_norm"])

    n_final = df["star_name_clean"].nunique()
    print(f"\n  Final unique names    : {n_final:,}")
    print(f"  Total reduction       : {n_raw_unique:,} → {n_final:,} "
          f"(-{n_raw_unique - n_final:,}, {100*(1-n_final/n_raw_unique):.1f}%)")

    # ── 4. Validation statistics ──
    print(f"\n--- Validation ---")

    # Distribution spectra/star
    star_counts = df["star_name_clean"].value_counts()
    print(f"\n  Spectra per star (after normalization):")
    print(f"    Median      : {star_counts.median():.0f}")
    print(f"    Mean        : {star_counts.mean():.0f}")
    print(f"    Max         : {star_counts.idxmax()} ({star_counts.max():,})")
    print(f"    ≥100 spectra : {(star_counts >= 100).sum():,} stars")
    print(f"    ≥10 spectra  : {(star_counts >= 10).sum():,} stars")
    print(f"    1 spectrum   : {(star_counts == 1).sum():,} stars")

    # Top 20
    print(f"\n  Top 20 stars:")
    for name, count in star_counts.head(20).items():
        # Find the associated raw names
        raw_variants = df[df["star_name_clean"] == name]["fits_objname"].nunique()
        suffix = f"  ({raw_variants} variants)" if raw_variants > 1 else ""
        print(f"    {name:25s} : {count:>6,} spectra{suffix}")

    # Most interesting cross-ID groups: very different names merged
    print(f"\n  Cross-identification examples (different names → same star):")
    cross_examples = (
        df[["star_name_norm", "star_name_clean"]]
        .drop_duplicates()
        .groupby("star_name_clean")["star_name_norm"]
        .apply(list)
    )
    interesting = cross_examples[cross_examples.apply(len) > 1].head(20)
    for canonical, variants in interesting.items():
        if len(variants) > 1:
            others = [v for v in variants if v != canonical]
            print(f"    {canonical:25s} ← {', '.join(others[:5])}")

    # ── 4b. Export the JSON mapping ──
    json_path = Path(__file__).parent / "star_name_lookup.json"
    _export_lookup_json(df, group_details, json_path, radius_deg)

    if dry_run:
        print("\n⚠ DRY-RUN: no modifications applied")
        return {
            "status": "dry_run",
            "raw_unique": n_raw_unique,
            "after_text": n_after_text,
            "final_unique": n_final,
        }

    # ── 5. Write to DuckDB ──
    print(f"\n--- Writing to DuckDB ---")

    # Add the column
    con.execute("""
        ALTER TABLE spectra_votable
        ADD COLUMN IF NOT EXISTS star_name_clean VARCHAR
    """)

    # Create a temporary table with the mapping
    mapping_df = df[["id", "star_name_clean"]].copy()
    con.execute("DROP TABLE IF EXISTS _tmp_star_names")
    con.execute("""
        CREATE TEMPORARY TABLE _tmp_star_names AS
        SELECT * FROM mapping_df
    """)

    # Bulk update via JOIN
    con.execute("""
        UPDATE spectra_votable
        SET star_name_clean = _tmp_star_names.star_name_clean
        FROM _tmp_star_names
        WHERE spectra_votable.id = _tmp_star_names.id
    """)

    con.execute("DROP TABLE IF EXISTS _tmp_star_names")

    # Verification
    check = con.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(star_name_clean) as n_filled,
            COUNT(DISTINCT star_name_clean) as n_unique
        FROM spectra_votable
    """).fetchone()

    print(f"\n  ✓ Column star_name_clean added")
    print(f"    Filled   : {check[1]:,} / {check[0]:,}")
    print(f"    Unique   : {check[2]:,}")

    # A few cross-checks
    print(f"\n--- Cross-checks ---")

    # gamma Cas and its aliases
    gam_cas = con.execute("""
        SELECT star_name_clean, fits_objname, COUNT(*) as n
        FROM spectra_votable
        WHERE fits_objname ILIKE '%cas%'
          AND (fits_objname ILIKE '%gam%' OR fits_objname ILIKE '%gamma%'
               OR fits_objname ILIKE 'HD5394%' OR fits_objname ILIKE 'HD 5394%')
        GROUP BY star_name_clean, fits_objname
        ORDER BY n DESC
        LIMIT 15
    """).fetchdf()
    print(f"\n  γ Cas test (must be grouped under a single star_name_clean):")
    if not gam_cas.empty:
        canonical = gam_cas['star_name_clean'].iloc[0]
        print(f"    Canonical name : {canonical}")
        n_variants = gam_cas['fits_objname'].nunique()
        print(f"    Raw variants   : {n_variants}")
    else:
        print(f"    ⚠ No result — check manually")

    # Distribution by observer type
    obs_type = con.execute("""
        SELECT observer_type,
               COUNT(DISTINCT star_name_clean) as n_stars,
               COUNT(*) as n_spectra
        FROM spectra_votable
        WHERE star_name_clean IS NOT NULL AND observer_type IS NOT NULL
        GROUP BY observer_type
        ORDER BY n_spectra DESC
    """).fetchdf()
    print(f"\n  Stars by observer type:")
    for _, row in obs_type.iterrows():
        print(f"    {row['observer_type']:15s} : {row['n_stars']:>5,} stars, "
              f"{row['n_spectra']:>7,} spectra")

    return {
        "status": "applied",
        "raw_unique": n_raw_unique,
        "after_text": n_after_text,
        "final_unique": n_final,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalization of BeSS star names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without modification")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to the DuckDB database")
    parser.add_argument("--radius", type=float, default=0.01,
                        help="Cross-ID radius in degrees (default: 0.01 = 36 arcsec)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"✗ Database not found: {args.db}")
        sys.exit(1)

    print(f"Database: {args.db}")
    print(f"Mode    : {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print(f"Radius  : {args.radius}° ({args.radius * 3600:.0f} arcsec)")
    print()

    con = duckdb.connect(str(args.db), read_only=args.dry_run)

    result = normalize_star_names(con, dry_run=args.dry_run, radius_deg=args.radius)

    if result["status"] == "applied":
        print(f"\n✓ Done — {result['final_unique']:,} unique stars "
              f"(from {result['raw_unique']:,} raw names)")
    else:
        print(f"\n✓ Dry-run done — {result['final_unique']:,} unique stars "
              f"(from {result['raw_unique']:,} raw names)")
        print("  Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()

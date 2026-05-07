#!/usr/bin/env python3
"""
03_validate_coordinates.py — Validation and normalization of geographic coordinates

Principle: keep raw values + add corrected columns
- fits_bss_lat  → site_lat_clean   (copy, already valid)
- fits_bss_long → site_lon_clean   (normalization to -180/+180)
- fits_bss_elev → site_elev_clean  (copy, already valid)

Applied corrections:
- Longitude: if > 180°, then lon - 360 (convention 0-360 → -180/+180)
- 561 IUE (satellite) spectra remain NULL for terrestrial coordinates

Usage:
    python 03_validate_coordinates.py [--dry-run] [--db PATH]
"""

import argparse
import sys
from pathlib import Path

import duckdb

# Default path to the database
DEFAULT_DB = Path(__file__).parent.parent.parent / "harvest" / "bess_data.duckdb"


def analyze_coordinates(con: duckdb.DuckDBPyConnection) -> dict:
    """Analyze the current state of the coordinates."""
    
    stats = con.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(fits_bss_lat) as has_lat,
            COUNT(fits_bss_long) as has_lon,
            COUNT(fits_bss_elev) as has_elev,
            COUNT(CASE WHEN fits_bss_long > 180 THEN 1 END) as lon_needs_fix,
            COUNT(CASE WHEN fits_bss_long < 0 THEN 1 END) as lon_already_negative,
            COUNT(CASE WHEN fits_bss_lat IS NULL OR fits_bss_long IS NULL THEN 1 END) as missing_coords
        FROM spectra_votable
    """).fetchone()
    
    return {
        "total": stats[0],
        "has_lat": stats[1],
        "has_lon": stats[2],
        "has_elev": stats[3],
        "lon_needs_fix": stats[4],
        "lon_already_negative": stats[5],
        "missing_coords": stats[6],
    }


def check_clean_columns_exist(con: duckdb.DuckDBPyConnection) -> bool:
    """Check whether the clean columns already exist."""
    cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'spectra_votable' 
        AND column_name IN ('site_lat_clean', 'site_lon_clean', 'site_elev_clean')
    """).fetchall()
    return len(cols) == 3


def add_clean_columns(con: duckdb.DuckDBPyConnection, dry_run: bool = False):
    """Add the clean columns if they do not exist."""
    
    if check_clean_columns_exist(con):
        print("✓ Clean columns already present")
        return
    
    if dry_run:
        print("[DRY-RUN] Would add columns: site_lat_clean, site_lon_clean, site_elev_clean")
        return
    
    print("Adding clean columns...")
    
    # Add the columns
    con.execute("ALTER TABLE spectra_votable ADD COLUMN IF NOT EXISTS site_lat_clean DOUBLE")
    con.execute("ALTER TABLE spectra_votable ADD COLUMN IF NOT EXISTS site_lon_clean DOUBLE")
    con.execute("ALTER TABLE spectra_votable ADD COLUMN IF NOT EXISTS site_elev_clean DOUBLE")
    
    print("✓ Columns added: site_lat_clean, site_lon_clean, site_elev_clean")


def apply_coordinate_corrections(con: duckdb.DuckDBPyConnection, dry_run: bool = False):
    """Apply the coordinate corrections."""
    
    if dry_run:
        print("[DRY-RUN] Would apply the corrections:")
        print("  - site_lat_clean = fits_bss_lat (copy)")
        print("  - site_lon_clean = CASE WHEN fits_bss_long > 180 THEN fits_bss_long - 360 ELSE fits_bss_long END")
        print("  - site_elev_clean = fits_bss_elev (copy)")
        return
    
    print("Applying corrections...")
    
    # Latitude: direct copy (already valid)
    con.execute("""
        UPDATE spectra_votable 
        SET site_lat_clean = fits_bss_lat
    """)
    
    # Longitude: normalize to -180/+180
    con.execute("""
        UPDATE spectra_votable 
        SET site_lon_clean = CASE 
            WHEN fits_bss_long > 180 THEN fits_bss_long - 360.0
            ELSE fits_bss_long
        END
    """)
    
    # Elevation: direct copy (already valid)
    con.execute("""
        UPDATE spectra_votable 
        SET site_elev_clean = fits_bss_elev
    """)
    
    print("✓ Corrections applied")


def validate_results(con: duckdb.DuckDBPyConnection):
    """Validate the results after correction."""
    
    print("\nValidation of results:")
    
    # Check ranges
    ranges = con.execute("""
        SELECT 
            MIN(site_lat_clean) as min_lat, MAX(site_lat_clean) as max_lat,
            MIN(site_lon_clean) as min_lon, MAX(site_lon_clean) as max_lon,
            MIN(site_elev_clean) as min_elev, MAX(site_elev_clean) as max_elev
        FROM spectra_votable
    """).fetchone()
    
    print(f"  Latitude  : [{ranges[0]:.2f}, {ranges[1]:.2f}]")
    print(f"  Longitude : [{ranges[2]:.2f}, {ranges[3]:.2f}]")
    print(f"  Elevation : [{ranges[4]:.0f}, {ranges[5]:.0f}] m")
    
    # Check that longitude is now normalized
    lon_issues = con.execute("""
        SELECT COUNT(*) 
        FROM spectra_votable 
        WHERE site_lon_clean > 180 OR site_lon_clean < -180
    """).fetchone()[0]
    
    if lon_issues > 0:
        print(f"  ⚠ {lon_issues} longitudes outside [-180, 180]")
    else:
        print("  ✓ All longitudes within [-180, 180]")
    
    # Count NULLs (IUE satellite)
    nulls = con.execute("""
        SELECT COUNT(*) 
        FROM spectra_votable 
        WHERE site_lat_clean IS NULL OR site_lon_clean IS NULL
    """).fetchone()[0]
    print(f"  ℹ {nulls} spectra without coordinates (IUE satellite)")


def main():
    parser = argparse.ArgumentParser(description="Validation and normalization of coordinates")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without modification")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to the DuckDB database")
    args = parser.parse_args()
    
    if not args.db.exists():
        print(f"✗ Database not found: {args.db}")
        sys.exit(1)
    
    print(f"Database: {args.db}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print()
    
    # Connection
    con = duckdb.connect(str(args.db), read_only=args.dry_run)
    
    # 1. Initial analysis
    print("=== Analysis of raw coordinates ===")
    stats = analyze_coordinates(con)
    print(f"  Total spectra: {stats['total']:,}")
    print(f"  With latitude : {stats['has_lat']:,}")
    print(f"  With longitude: {stats['has_lon']:,}")
    print(f"  With elevation: {stats['has_elev']:,}")
    print(f"  Longitudes to fix (> 180°): {stats['lon_needs_fix']:,}")
    print(f"  Longitudes already negative: {stats['lon_already_negative']:,}")
    print(f"  Without coordinates (IUE): {stats['missing_coords']:,}")
    print()
    
    # 2. Add columns
    print("=== Adding clean columns ===")
    add_clean_columns(con, dry_run=args.dry_run)
    print()
    
    # 3. Apply corrections
    print("=== Applying corrections ===")
    apply_coordinate_corrections(con, dry_run=args.dry_run)
    
    # 4. Validation
    if not args.dry_run:
        validate_results(con)
    
    print("\n✓ Done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
verify_artifacts.py — Sanity checks for the public BESS-Bench release.

Usage:
    python tools/verify_artifacts.py [RELEASE_DIR]

With no argument, verifies the current repository (useful in dev).
With an argument, verifies the release in RELEASE_DIR.

Returns 0 if everything is OK, 1 otherwise.
"""

import sys
import json
import hashlib
import csv
from pathlib import Path

# ── Expected config (taken from the NeurIPS 2026 paper) ──────────────────────────
EXPECTED = {
    "n_stars":        1468,
    "n_spectra":      339_115,
    "n_train_stars":  1024,
    "n_val_stars":    219,
    "n_test_stars":   225,
    "n_train_spec":   241_057,
    "n_val_spec":     44_383,
    "n_test_spec":    53_675,
    "anon_hf_dataset": "anonym-submit-26/bess-bench-26",
    "anon_hf_model":   "anonym-submit-26/bemae-halpha-v1",
    "anon_github":     "anonym-submit-26",
}

IDENTITY_PATTERNS = ["/share/home", "/home/", "/Users/"]

REQUIRED_FILES = [
    "README.md",
    "dataset/DATASHEET.md",
    "LICENSE",
    "REPRODUCE.md",
    "requirements.txt",
    "dataset/croissant.json",
    "dataset/splits.csv",
    "dataset/star_list.csv",
    "dataset/README.md",
]

ERRORS = []
WARNINGS = []


def err(msg):
    ERRORS.append(f"  ❌ {msg}")


def warn(msg):
    WARNINGS.append(f"  ⚠️  {msg}")


def ok(msg):
    print(f"  ✅ {msg}")


# ── Chemin racine ──────────────────────────────────────────────────────────────
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
print(f"\n🔍 Verifying: {ROOT}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fichiers requis
# ─────────────────────────────────────────────────────────────────────────────
print("── 1. Fichiers requis ─────────────────────────────────────────────────")
for rel in REQUIRED_FILES:
    p = ROOT / rel
    if p.exists():
        ok(rel)
    else:
        err(f"Missing file: {rel}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. splits.csv — train/val/test counts and MD5 formula
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. splits.csv ──────────────────────────────────────────────────────")
splits_path = ROOT / "dataset/splits.csv"
if splits_path.exists():
    counts = {}
    stars_split = {}
    with open(splits_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            counts[split] = counts.get(split, 0) + 1
            stars_split[row["star_name"]] = split

    for split_name, key in [("train", "n_train_stars"), ("validation", "n_val_stars"), ("test", "n_test_stars")]:
        got = counts.get(split_name, 0)
        exp = EXPECTED[key]
        if got == exp:
            ok(f"{split_name}: {got} stars")
        else:
            err(f"{split_name}: expected {exp}, found {got}")

    total = sum(counts.values())
    if total == EXPECTED["n_stars"]:
        ok(f"Total: {total} stars")
    else:
        err(f"Total: expected {EXPECTED['n_stars']}, found {total}")

    # Check the MD5 formula
    mismatches = 0
    for star, split in stars_split.items():
        h = int(hashlib.md5(f"bess_bench_v1::{star}".encode()).hexdigest()[:8], 16) % 100
        expected_split = "train" if h < 70 else "validation" if h < 85 else "test"
        if split != expected_split:
            mismatches += 1
    if mismatches == 0:
        ok("MD5 formula reproduces the split with 0 errors")
    else:
        err(f"MD5 formula: {mismatches}/{total} disagreements")


# ─────────────────────────────────────────────────────────────────────────────
# 3. star_list.csv — total n_spectra and split consistency
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. star_list.csv ───────────────────────────────────────────────────")
star_list_path = ROOT / "dataset/star_list.csv"
if star_list_path.exists():
    total_spec = 0
    spec_per_split = {}
    star_list_splits = {}
    with open(star_list_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n_spectra"])
            total_spec += n
            split = row["split"]
            spec_per_split[split] = spec_per_split.get(split, 0) + n
            star_list_splits[row["star_name"]] = split

    if total_spec == EXPECTED["n_spectra"]:
        ok(f"Total n_spectra: {total_spec}")
    else:
        err(f"Total n_spectra: expected {EXPECTED['n_spectra']}, found {total_spec}")

    for split_name, key in [("train", "n_train_spec"), ("validation", "n_val_spec"), ("test", "n_test_spec")]:
        got = spec_per_split.get(split_name, 0)
        exp = EXPECTED[key]
        if got == exp:
            ok(f"n_spectra {split_name}: {got}")
        else:
            err(f"n_spectra {split_name}: expected {exp}, found {got}")

    # Consistency with splits.csv
    if splits_path.exists():
        mismatch = sum(1 for k in stars_split if stars_split.get(k) != star_list_splits.get(k))
        if mismatch == 0:
            ok("Splits consistent between splits.csv and star_list.csv")
        else:
            err(f"{mismatch} stars have different splits between splits.csv and star_list.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Anonymous identity — leak scan
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Identity leak scan ──────────────────────────────────────")
TEXT_EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".tex",
             ".txt", ".toml", ".cfg", ".slurm", ".rst", ""}
leak_count = 0
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    if ".git" in p.parts or ".venv" in p.parts or "__pycache__" in p.parts:
        continue
    # tools/ legitimately contains the patterns to detect them
    if "tools" in p.parts:
        continue
    if p.suffix not in TEXT_EXTS and p.name != "LICENSE":
        continue
    try:
        text = p.read_text(errors="ignore").lower()
    except Exception:
        continue
    for pat in IDENTITY_PATTERNS:
        if pat in text:
            err(f"Leak '{pat}' in {p.relative_to(ROOT)}")
            leak_count += 1
            break

if leak_count == 0:
    ok(f"No leak detected across {sum(1 for _ in ROOT.rglob('*') if _.is_file())} files")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Verify that anonymous IDs are present
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Presence of anonymous IDs ────────────────────────────────────────")
# README_HF must contain the anonymous HF ID
readme_hf = ROOT / "dataset/README.md"
if readme_hf.exists():
    text = readme_hf.read_text()
    if EXPECTED["anon_hf_dataset"] in text:
        ok(f"README_HF contains {EXPECTED['anon_hf_dataset']}")
    else:
        err(f"README_HF does not contain {EXPECTED['anon_hf_dataset']}")

# croissant.json must contain the anonymous URL
croissant = ROOT / "dataset/croissant.json"
if croissant.exists():
    data = json.loads(croissant.read_text())
    url = data.get("url", "")
    if EXPECTED["anon_hf_dataset"] in url:
        ok(f"croissant.json url contains {EXPECTED['anon_hf_dataset']}")
    else:
        err(f"croissant.json url: '{url}' does not contain {EXPECTED['anon_hf_dataset']}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
if WARNINGS:
    for w in WARNINGS:
        print(w)

if ERRORS:
    print(f"\n🔴 {len(ERRORS)} error(s):")
    for e in ERRORS:
        print(e)
    print()
    sys.exit(1)
else:
    print(f" ✅  All checks pass — release ready.")
    print()
    sys.exit(0)

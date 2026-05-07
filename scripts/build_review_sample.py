"""Build a public reviewer sample of BESS-Bench v1.0 (~1% of the corpus).

Method
------
- Load the full HF dataset (single 'train' split, 339,115 rows).
- Load star-level splits.csv (1,024 train / 219 val / 225 test stars).
- For each split, draw N=50 stars uniformly at random (seed=42) and
  retain at most 20 spectra per star. Union -> ~3,000 spectra.
- Add a `split` column so reviewers can re-create the per-split slices.
- Push as a separate dataset 'anonym-submit-26/bess-bench-26-sample'.

The schema, dtypes and column names are identical to the full release.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT.parent / "anon_admin" / "hf_anonym.env"
DATASET_DIR = Path(os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset_v1.0"))
SPLITS_CSV = ROOT / "dataset" / "splits.csv"
REPO_ID = "anonym-submit-26/bess-bench-26-sample"
SEED = 42
N_STARS_PER_SPLIT = 50
MAX_SPECTRA_PER_STAR = 20


def load_token() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"HF_TOKEN not found in {ENV_FILE}")


def load_split_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(SPLITS_CSV) as f:
        next(f)
        for line in f:
            star, split = line.rstrip("\n").rsplit(",", 1)
            mapping[star] = split
    return mapping


def main() -> None:
    token = load_token()
    login(token=token, add_to_git_credential=False)
    rng = np.random.default_rng(SEED)

    print(f"[1/5] Loading {DATASET_DIR} ...")
    ds = load_from_disk(str(DATASET_DIR))["train"]
    print(f"      total rows: {len(ds):,}")

    print("[2/5] Loading splits.csv ...")
    star_to_split = load_split_map()
    print(f"      stars in splits.csv: {len(star_to_split):,}")

    star_names = ds["star_name"]
    by_split_stars: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for star, split in star_to_split.items():
        by_split_stars[split].append(star)
    for split, stars in by_split_stars.items():
        print(f"      {split}: {len(stars)} stars")

    print(f"[3/5] Sampling {N_STARS_PER_SPLIT} stars per split (seed={SEED}) ...")
    selected_stars: dict[str, set[str]] = {}
    for split, stars in by_split_stars.items():
        n = min(N_STARS_PER_SPLIT, len(stars))
        picked = rng.choice(np.array(sorted(stars)), size=n, replace=False)
        selected_stars[split] = set(picked.tolist())
        print(f"      {split}: picked {n} stars")

    print("[4/5] Selecting indices (cap per star = "
          f"{MAX_SPECTRA_PER_STAR}) ...")
    counters: dict[str, int] = {}
    keep_indices: list[int] = []
    keep_split: list[str] = []
    name_arr = np.asarray(star_names)
    flat = {s: split for split, stars in selected_stars.items() for s in stars}
    for i, name in enumerate(name_arr):
        split = flat.get(name)
        if split is None:
            continue
        c = counters.get(name, 0)
        if c >= MAX_SPECTRA_PER_STAR:
            continue
        counters[name] = c + 1
        keep_indices.append(i)
        keep_split.append(split)
    print(f"      retained: {len(keep_indices):,} spectra")

    sample = ds.select(keep_indices)
    sample = sample.add_column("split", keep_split)
    print(f"      sample columns: {len(sample.column_names)}")

    print(f"[5/5] Pushing to {REPO_ID} ...")
    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    sample.push_to_hub(REPO_ID, token=token, max_shard_size="500MB")
    print("      done.")


if __name__ == "__main__":
    main()

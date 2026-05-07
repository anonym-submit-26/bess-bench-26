#!/usr/bin/env python3
"""
generate_croissant.py — Create a Croissant 1.0 descriptor (with RAI fields)
for BESS-Bench.

This script writes `dataset/croissant.json` which can be placed as-is in the
HuggingFace repo. URLs are configurable via --repo-id (useful to switch
between the anonymous review mirror and the final post-acceptance mirror).

Spec : https://github.com/mlcommons/croissant
RAI extension : https://github.com/mlcommons/croissant/blob/main/docs/rai-spec.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build(repo_id: str, star_list_csv: Path) -> dict:
    base = f"https://huggingface.co/datasets/{repo_id}"
    resolve = f"{base}/resolve/main"

    stars_df = pd.read_csv(star_list_csv)
    n_stars = int(len(stars_df))
    n_spectra = int(stars_df["n_spectra"].sum())

    # ── Core Croissant document ──
    doc = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "BESS-Bench",
        "description": (
            "Long-baseline, mixed-quality Be-star spectroscopic time-series benchmark. "
            f"{n_spectra} spectra across {n_stars} unique Be stars spanning 1990-2025, "
            "with 81.7% amateur observers at the row level (91.6% at the physical-"
            "observation level). Bundles three benchmarked downstream tasks "
            "(SpecProbe Halpha feature regression, LineTransfer Hbeta-to-Halpha "
            "generalisation, EWForecast short-horizon EW(Halpha) forecasting), "
            "fixed per-star splits, and classical baselines."
        ),
        "version": "1.0.0",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": base,
        "keywords": [
            "astronomy", "astrophysics", "spectroscopy", "Be stars",
            "stellar spectra", "time series", "representation learning",
            "forecasting", "benchmark", "citizen science",
        ],
        "citeAs": (
            "@misc{bess_bench_2026, title={BESS-Bench: Long-baseline Be-star spectra "
            "for ML time-series research}, author={Anonymous}, year={2026}, "
            f"howpublished={{\\url{{{base}}}}}}}"
        ),
        "creator": {
            "@type": "Person",
            "name": "Anonymous (NeurIPS 2026 D&B Track review)",
        },
        "publisher": {
            "@type": "Organization",
            "name": "Anonymous (NeurIPS 2026 D&B Track review)",
        },
        "datePublished": "2026-05-01",
        "isLiveDataset": False,

        # ── Distribution ──
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "hf-dataset-repo",
                "name": "hf-dataset-repo",
                "description": (
                    "HuggingFace dataset repository hosting the 21-shard Parquet "
                    "release and this Croissant descriptor."
                ),
                "contentUrl": base,
                "encodingFormat": "git+https",
                "sha256": "main",
            },
            {
                "@type": "cr:FileSet",
                "@id": "parquet-shards",
                "name": "parquet-shards",
                "description": "Arrow/Parquet shards produced by `datasets.save_to_disk`.",
                "containedIn": {"@id": "hf-dataset-repo"},
                "encodingFormat": "application/vnd.apache.parquet",
                "includes": "data/train-*.parquet",
            },
            {
                "@type": "cr:FileObject",
                "@id": "splits-csv",
                "name": "splits-csv",
                "description": "Deterministic per-star split assignment.",
                "containedIn": {"@id": "hf-dataset-repo"},
                "contentUrl": f"{resolve}/splits.csv",
                "encodingFormat": "text/csv",
            },
            {
                "@type": "cr:FileObject",
                "@id": "star-list-csv",
                "name": "star-list-csv",
                "description": "One row per star with aggregates (n_spectra, coords, MJD range, amateur fraction).",
                "containedIn": {"@id": "hf-dataset-repo"},
                "contentUrl": f"{resolve}/star_list.csv",
                "encodingFormat": "text/csv",
            },
        ],

        # ── Record sets ──
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "spectra",
                "name": "spectra",
                "description": "One row per spectrum; arrays bundled inline.",
                "field": [
                    {"@type": "cr:Field", "@id": "spectra/spectrum_id",
                     "name": "spectrum_id", "dataType": "sc:Text",
                     "description": "Unique MD5 identifier.",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "spectrum_id"}}},
                    {"@type": "cr:Field", "@id": "spectra/star_name",
                     "name": "star_name", "dataType": "sc:Text",
                     "description": "Canonical de-duplicated star name.",
                     "references": {"field": {"@id": "stars/star_name"}},
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "star_name"}}},
                    {"@type": "cr:Field", "@id": "spectra/wavelength",
                     "name": "wavelength", "dataType": "sc:Float",
                     "repeated": True,
                     "description": "Native wavelength grid (Angstrom).",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "wavelength"}}},
                    {"@type": "cr:Field", "@id": "spectra/flux",
                     "name": "flux", "dataType": "sc:Float", "repeated": True,
                     "description": "Calibrated flux array.",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "flux"}}},
                    {"@type": "cr:Field", "@id": "spectra/flux_error",
                     "name": "flux_error", "dataType": "sc:Float", "repeated": True,
                     "description": "Flux uncertainty (nullable).",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "flux_error"}}},
                    {"@type": "cr:Field", "@id": "spectra/mjd",
                     "name": "mjd", "dataType": "sc:Float",
                     "description": "Modified Julian Date.",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "mjd"}}},
                    {"@type": "cr:Field", "@id": "spectra/snr",
                     "name": "snr", "dataType": "sc:Float",
                     "description": "DER_SNR.",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "snr"}}},
                    {"@type": "cr:Field", "@id": "spectra/observer_type",
                     "name": "observer_type", "dataType": "sc:Text",
                     "description": "amateur / professional / unknown.",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "observer_type"}}},
                    {"@type": "cr:Field", "@id": "spectra/is_echelle_order",
                     "name": "is_echelle_order", "dataType": "sc:Boolean",
                     "description": "True when this row is an individual echelle order.",
                     "source": {"fileSet": {"@id": "parquet-shards"},
                                "extract": {"column": "is_echelle_order"}}},
                ],
                "key": {"@id": "spectra/spectrum_id"},
            },
            {
                "@type": "cr:RecordSet",
                "@id": "stars",
                "name": "stars",
                "description": "Per-star aggregates and official split assignment.",
                "field": [
                    {"@type": "cr:Field", "@id": "stars/star_name",
                     "name": "star_name", "dataType": "sc:Text",
                     "source": {"fileObject": {"@id": "star-list-csv"},
                                "extract": {"column": "star_name"}}},
                    {"@type": "cr:Field", "@id": "stars/n_spectra",
                     "name": "n_spectra", "dataType": "sc:Integer",
                     "source": {"fileObject": {"@id": "star-list-csv"},
                                "extract": {"column": "n_spectra"}}},
                    {"@type": "cr:Field", "@id": "stars/split",
                     "name": "split", "dataType": "sc:Text",
                     "description": "train / validation / test (by-star).",
                     "source": {"fileObject": {"@id": "star-list-csv"},
                                "extract": {"column": "split"}}},
                    {"@type": "cr:Field", "@id": "stars/ra_deg",
                     "name": "ra_deg", "dataType": "sc:Float",
                     "source": {"fileObject": {"@id": "star-list-csv"},
                                "extract": {"column": "ra_deg"}}},
                    {"@type": "cr:Field", "@id": "stars/dec_deg",
                     "name": "dec_deg", "dataType": "sc:Float",
                     "source": {"fileObject": {"@id": "star-list-csv"},
                                "extract": {"column": "dec_deg"}}},
                ],
                "key": {"@id": "stars/star_name"},
            },
        ],

        # ── RAI fields (Croissant RAI extension) ──
        "rai:dataCollection": (
            "Spectra were uploaded voluntarily by amateur and professional "
            "observers to the BeSS public archive (LESIA, Observatoire de Paris) "
            "between 1990 and 2025. We harvested them via the public SSA/VO "
            "protocol; curation consisted of quality gating, regex-based "
            "instrument normalisation, spatial+textual star-name deduplication "
            "(4755 raw names -> 1468 canonical stars), and schema export to "
            "HuggingFace Parquet."
        ),
        "rai:dataCollectionType": ["Archive", "Citizen science"],
        "rai:dataCollectionRawData": (
            "Raw FITS files remain on the BeSS archive at http://basebe.obspm.fr "
            "and can be retrieved by any researcher with a free account. This "
            "dataset redistributes only the cleaned, ML-ready export."
        ),
        "rai:dataCollectionTimeframe": "1990-03-11 / 2025-12-30",
        "rai:dataImputationProtocol": (
            "flux_error is left null when absent from FITS (no imputation). "
            "spectral_resolution falls back to a per-spectrograph median only "
            "when the header value is missing; the fallback is flagged via "
            "instrument_confidence < 1."
        ),
        "rai:dataPreprocessingProtocol": [
            "SNR estimation via DER_SNR (Stoehr 2008) and continuum windows.",
            "Spectrograph/telescope normalisation via regex parsers (99.2 % "
            "mean confidence).",
            "Coordinate validation and longitude normalisation.",
            "Star-name deduplication (textual Levenshtein + 10-arcsec SIMBAD "
            "cross-match).",
            "Echelle-order flagging (spectrograph class + structural checks).",
            "Deterministic per-star split via md5('bess_bench_v1::'+star_name).",
        ],
        "rai:dataAnnotationProtocol": (
            "All v1.0 scored task labels (SpecProbe, LineTransfer, EWForecast) "
            "are computed automatically from the spectra by "
            "downstream/compute_spectral_features.py. Halpha morphological "
            "classes (single-peak, double-peak, shell, pure-absorption) are "
            "derived programmatically from line measurements as dataset-level "
            "metadata, not as a scored benchmark task in v1.0. No manual "
            "annotation and no crowdworkers were involved in v1.0."
        ),
        "rai:dataAnnotationAnalysis": (
            "All v1.0 scored task labels are derived automatically from spectral "
            "measurements; inter-annotator agreement is therefore not applicable."
        ),
        "rai:dataAnnotationPlatform": "Internal Jupyter tooling; no third-party platform.",
        "rai:dataUseCases": [
            "Self-supervised representation learning on stellar spectra.",
            "Short-horizon temporal forecasting under persistence baselines.",
            "Cross-line generalisation studies on Balmer-series profiles.",
            "Cross-instrument normalisation studies (amateur vs professional).",
        ],
        "rai:dataLimitations": [
            "Northern-hemisphere bias (dec > -30 deg dominated).",
            "Brightness bias toward V < 8 Be stars.",
            "Heavy re-use of a few popular amateur spectrographs (LHIRES III, "
            "eShel).",
            "Oversampling of a handful of historically active targets "
            "(gamma Cas, zeta Tau, delta Sco).",
            "flux_error is null for ~55 % of spectra.",
        ],
        "rai:dataSocialImpact": (
            "BESS-Bench formalises 35 years of volunteer astronomical labour "
            "into a reproducible ML benchmark, giving credit to the amateur "
            "community and enabling cross-disciplinary studies of "
            "citizen-science data quality."
        ),
        "rai:dataBiases": [
            "Geographic bias (northern hemisphere)",
            "Instrument bias (popular amateur spectrographs oversampled)",
            "Target bias (famous variable Be stars oversampled)",
        ],
        "rai:personalSensitiveInformation": (
            "No personally identifiable information is distributed. FITS "
            "OBSERVER keywords are stripped at export. Site coordinates "
            "(latitude, longitude) are rounded to 0.1 deg (approx. 11 km) "
            "and site elevation is rounded to the nearest 100 m to prevent "
            "observer re-identification while preserving statistical "
            "utility (hemispheric bias, time-zone effects, air-mass / "
            "extinction analyses)."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Annual v1.x minor releases integrating new BeSS harvests and bug "
            "fixes. Versions pinned by Git tag and HuggingFace revision. "
            "Existing splits are immutable within v1.x to preserve benchmark "
            "comparability. Opt-out requests from observers are honoured "
            "within 30 days in the next minor release."
        ),
    }
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="ANONYMOUS/bess-bench",
                    help="HuggingFace repo id")
    ap.add_argument("--star-list", type=Path,
                    default=Path(__file__).resolve().parent.parent / "paper" / "star_list.csv")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "paper" / "croissant.json")
    args = ap.parse_args()

    doc = build(args.repo_id, args.star_list)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"[+] {args.out}")
    print(f"    repo-id: {args.repo_id}")
    print(f"    fields:  {sum(len(rs['field']) for rs in doc['recordSet'])} across "
          f"{len(doc['recordSet'])} recordSets")
    print(f"    RAI keys: {sum(1 for k in doc if k.startswith('rai:'))}")


if __name__ == "__main__":
    main()

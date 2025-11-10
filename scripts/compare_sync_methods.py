#!/usr/bin/env python3
"""
Compare synchronization results across different methods:
- Hybrid (full-signal first, chunking if r<0.7)
- Chunked-only (always use chunking)
- Original (if available)

Generates summary statistics and identifies improvements.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


def load_sync_params(sync_file: Path) -> Dict:
    """Load sync_params.json file."""
    with open(sync_file) as f:
        return json.load(f)


def extract_key_metrics(params: Dict) -> Dict:
    """Extract key metrics from sync params."""
    qc = params.get("qc_metrics", {})

    # Get alignment method from qc_metrics (newer format) or params (older format)
    alignment_method = qc.get("alignment_method", params.get("alignment_method", "unknown"))

    # Get correlation - try multiple field names
    correlation = qc.get("peak_correlation",
                        qc.get("full_signal_correlation",
                               params.get("initial_correlation", 0.0)))

    return {
        "median_error_ms": qc.get("median_error_ms", float("inf")),
        "max_error_ms": qc.get("max_error_ms", float("inf")),
        "drift_rate_ppm": qc.get("drift_rate_ppm", 0.0),
        "sync_quality": qc.get("sync_quality", "unknown"),
        "n_windows": qc.get("n_windows", 0),
        "alignment_method": alignment_method,
        "correlation": correlation,
    }


def compare_methods(base_dir: Path):
    """Compare sync methods across all recordings."""

    sync_dir = base_dir / "outputs" / "sync"

    # Find all current sync_params.json (hybrid results)
    hybrid_files = sorted(sync_dir.glob("*/run-*/sync_params.json"))

    # Find all chunked-only backups
    chunked_files = sorted(sync_dir.glob("*/run-*/sync_params_chunked.json"))

    print("=" * 80)
    print("SYNCHRONIZATION METHOD COMPARISON")
    print("=" * 80)
    print(f"\nFound {len(hybrid_files)} hybrid sync results")
    print(f"Found {len(chunked_files)} chunked-only backup results")
    print()

    # Load all results
    hybrid_results = {}
    chunked_results = {}

    for f in hybrid_files:
        key = f"{f.parent.parent.name}/{f.parent.name}"
        hybrid_results[key] = load_sync_params(f)

    for f in chunked_files:
        key = f"{f.parent.parent.name}/{f.parent.name}"
        chunked_results[key] = load_sync_params(f)

    # Compare recordings that have both hybrid and chunked results
    common_keys = set(hybrid_results.keys()) & set(chunked_results.keys())

    if not common_keys:
        print("No recordings with both hybrid and chunked-only results found.")
        print("This is expected if cleanup script removed backup files.")
        return

    print(f"Comparing {len(common_keys)} recordings with both methods...\n")

    # Statistics
    stats = {
        "hybrid_better": [],
        "chunked_better": [],
        "equal": [],
        "hybrid_used_full": [],
        "hybrid_used_chunked": [],
    }

    improvements = []

    for key in sorted(common_keys):
        h_metrics = extract_key_metrics(hybrid_results[key])
        c_metrics = extract_key_metrics(chunked_results[key])

        # Track which method hybrid used
        if h_metrics["alignment_method"] == "full_signal":
            stats["hybrid_used_full"].append(key)
        elif h_metrics["alignment_method"] == "chunked":
            stats["hybrid_used_chunked"].append(key)

        # Compare median error
        h_err = h_metrics["median_error_ms"]
        c_err = c_metrics["median_error_ms"]

        if h_err < c_err * 0.95:  # At least 5% improvement
            stats["hybrid_better"].append(key)
            improvements.append({
                "recording": key,
                "hybrid_error": h_err,
                "chunked_error": c_err,
                "improvement_pct": (c_err - h_err) / c_err * 100,
                "method": h_metrics["alignment_method"],
            })
        elif c_err < h_err * 0.95:
            stats["chunked_better"].append(key)
        else:
            stats["equal"].append(key)

    # Print summary
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nHybrid approach method usage:")
    print(f"  Full-signal method: {len(stats['hybrid_used_full'])} recordings")
    print(f"  Chunked method:     {len(stats['hybrid_used_chunked'])} recordings")
    print()

    print(f"Accuracy comparison (hybrid vs chunked-only):")
    print(f"  Hybrid better:      {len(stats['hybrid_better'])} recordings")
    print(f"  Chunked-only better: {len(stats['chunked_better'])} recordings")
    print(f"  Approximately equal: {len(stats['equal'])} recordings")
    print()

    # Print recordings where hybrid was better
    if improvements:
        print("=" * 80)
        print(f"RECORDINGS WHERE HYBRID IMPROVED ACCURACY ({len(improvements)} total)")
        print("=" * 80)
        print(f"\n{'Recording':<25} {'Method':<12} {'Hybrid (ms)':<12} {'Chunked (ms)':<12} {'Improvement':<12}")
        print("-" * 80)

        for imp in sorted(improvements, key=lambda x: x["improvement_pct"], reverse=True)[:20]:
            print(f"{imp['recording']:<25} {imp['method']:<12} "
                  f"{imp['hybrid_error']:<12.2f} {imp['chunked_error']:<12.2f} "
                  f"{imp['improvement_pct']:<12.1f}%")

    # Print overall metrics
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY METRICS")
    print("=" * 80)

    h_errors = [extract_key_metrics(hybrid_results[k])["median_error_ms"]
                for k in common_keys]
    c_errors = [extract_key_metrics(chunked_results[k])["median_error_ms"]
                for k in common_keys]

    print(f"\nHybrid approach:")
    print(f"  Mean median error:   {np.mean(h_errors):.2f} ms")
    print(f"  Median median error: {np.median(h_errors):.2f} ms")
    print(f"  Std median error:    {np.std(h_errors):.2f} ms")
    print(f"  Max median error:    {np.max(h_errors):.2f} ms")

    print(f"\nChunked-only approach:")
    print(f"  Mean median error:   {np.mean(c_errors):.2f} ms")
    print(f"  Median median error: {np.median(c_errors):.2f} ms")
    print(f"  Std median error:    {np.std(c_errors):.2f} ms")
    print(f"  Max median error:    {np.max(c_errors):.2f} ms")

    print(f"\nOverall improvement:")
    mean_improvement = np.mean(c_errors) - np.mean(h_errors)
    print(f"  Mean error reduction: {mean_improvement:.2f} ms ({mean_improvement/np.mean(c_errors)*100:.1f}%)")

    print("\n" + "=" * 80)


def analyze_current_results(base_dir: Path):
    """Analyze current hybrid synchronization results."""

    sync_dir = base_dir / "outputs" / "sync"
    sync_files = sorted(sync_dir.glob("*/run-*/sync_params.json"))

    print("\n" + "=" * 80)
    print("CURRENT HYBRID SYNCHRONIZATION RESULTS")
    print("=" * 80)
    print(f"\nTotal recordings: {len(sync_files)}\n")

    # Collect metrics
    method_counts = defaultdict(int)
    quality_counts = defaultdict(int)
    all_errors = []
    excellent_recordings = []

    for sync_file in sync_files:
        params = load_sync_params(sync_file)
        metrics = extract_key_metrics(params)

        method_counts[metrics["alignment_method"]] += 1
        quality_counts[metrics["sync_quality"]] += 1
        all_errors.append(metrics["median_error_ms"])

        # Track excellent recordings (< 10ms median error)
        if metrics["median_error_ms"] < 10.0:
            key = f"{sync_file.parent.parent.name}/{sync_file.parent.name}"
            excellent_recordings.append((key, metrics["median_error_ms"], metrics["alignment_method"]))

    # Print method distribution
    print("Method distribution:")
    for method, count in sorted(method_counts.items()):
        pct = count / len(sync_files) * 100
        print(f"  {method:<20}: {count:>3} recordings ({pct:>5.1f}%)")
    print()

    # Print quality distribution
    print("Sync quality distribution:")
    for quality, count in sorted(quality_counts.items()):
        pct = count / len(sync_files) * 100
        print(f"  {quality.upper():<20}: {count:>3} recordings ({pct:>5.1f}%)")
    print()

    # Print error statistics
    print("Median alignment error statistics:")
    print(f"  Mean:   {np.mean(all_errors):.2f} ms")
    print(f"  Median: {np.median(all_errors):.2f} ms")
    print(f"  Std:    {np.std(all_errors):.2f} ms")
    print(f"  Min:    {np.min(all_errors):.2f} ms")
    print(f"  Max:    {np.max(all_errors):.2f} ms")
    print()

    # Print excellent recordings
    print(f"Excellent recordings (< 10 ms median error): {len(excellent_recordings)}")
    if excellent_recordings:
        print(f"\nTop 10 most accurate:")
        print(f"{'Recording':<25} {'Error (ms)':<12} {'Method':<12}")
        print("-" * 50)
        for rec, err, method in sorted(excellent_recordings, key=lambda x: x[1])[:10]:
            print(f"{rec:<25} {err:<12.2f} {method:<12}")

    print("\n" + "=" * 80)


def main():
    base_dir = Path("/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline")

    # First analyze current hybrid results
    analyze_current_results(base_dir)

    # Then compare with chunked-only if backups exist
    try:
        compare_methods(base_dir)
    except Exception as e:
        print(f"\nNote: Could not compare with chunked-only backups: {e}")
        print("This is expected if cleanup script has already removed backup files.")

    print("\nDONE")


if __name__ == "__main__":
    main()

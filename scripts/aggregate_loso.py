"""
Aggregate leave-one-subject-out results across experiments.

Scans outputs/evaluation for per-fold metrics written by
scripts/run_loso_experiment.py and reports, per experiment:
mean / worst-subject PCK and MPJPE, classification accuracy, and the
sitting/standing breakdown. Writes outputs/evaluation/loso_summary.json.

Usage:
    python scripts/aggregate_loso.py
    python scripts/aggregate_loso.py --experiments baseline3c rootrel_geo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

ALL_SUBJECTS = ("001", "002", "003", "004", "005", "006", "007")

DEFAULT_EXPERIMENTS = (
    "baseline3c",
    "rootrel",
    "rootrel_geo",
    "geo_only",
    "rootrel_geo_aug",
    "rootrel_geo_aug_adv",
    "rootrel_gcn_aug",
)


def load_fold(experiment, subject, adapt_norm=False):
    suffix = "_adaptnorm_metrics.json" if adapt_norm else "_metrics.json"
    path = (
        config.EVALUATION_DIR
        / f"{config.MODEL_NAME}_{experiment}_test{subject}{suffix}"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_by_pose(experiment, subject):
    path = (
        config.EVALUATION_DIR
        / f"{config.MODEL_NAME}_{experiment}_test{subject}_test_metrics_by_pose.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(experiment, adapt_norm=False):
    folds = {}
    for subject in ALL_SUBJECTS:
        metrics = load_fold(experiment, subject, adapt_norm=adapt_norm)
        if metrics is None:
            continue

        fold = {
            "pck": metrics.get("normalized_pck"),
            "mpjpe": metrics.get("normalized_mpjpe"),
            "pixel_mpjpe": metrics.get("pixel_mpjpe"),
            "classification_accuracy": metrics.get(
                "pose_classification_accuracy"
            ),
        }

        by_pose = load_by_pose(experiment, subject)
        if by_pose is not None:
            for pose in ("sitting", "standing"):
                entry = by_pose.get("results", {}).get(pose)
                if entry:
                    fold[f"{pose}_pck"] = entry.get("normalized_pck")
                    fold[f"{pose}_mpjpe"] = entry.get("normalized_mpjpe")

        folds[subject] = fold

    if not folds:
        return None

    def collect(key):
        values = [
            fold[key]
            for fold in folds.values()
            if fold.get(key) is not None
        ]
        return values

    pcks = collect("pck")
    summary = {
        "folds_completed": len(folds),
        "per_fold": folds,
        "mean_pck": sum(pcks) / len(pcks) if pcks else None,
        "worst_pck": min(pcks) if pcks else None,
        "worst_subject": (
            min(folds, key=lambda s: folds[s]["pck"] or 0.0) if pcks else None
        ),
    }

    for key in (
        "mpjpe",
        "classification_accuracy",
        "sitting_pck",
        "standing_pck",
        "sitting_mpjpe",
        "standing_mpjpe",
    ):
        values = collect(key)
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)
            if "pck" in key or key == "classification_accuracy":
                summary[f"worst_{key}"] = min(values)
            else:
                summary[f"worst_{key}"] = max(values)

    return summary


def format_value(value, fmt="{:.3f}"):
    return "  -  " if value is None else fmt.format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument("--adapt-norm", action="store_true")
    args = parser.parse_args()

    all_summaries = {}
    for experiment in args.experiments:
        summary = summarize(experiment, adapt_norm=args.adapt_norm)
        if summary is not None:
            all_summaries[experiment] = summary

    if not all_summaries:
        print("No completed folds found for the requested experiments.")
        return

    header = (
        f"{'experiment':22s} {'folds':>5s} {'meanPCK':>8s} {'worstPCK':>9s} "
        f"{'worst_subj':>10s} {'sitPCK':>7s} {'standPCK':>9s} "
        f"{'MPJPE':>7s} {'clsAcc':>7s}"
    )
    print("\nLOSO SUMMARY" + (" (adapt-norm)" if args.adapt_norm else ""))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for experiment, s in all_summaries.items():
        print(
            f"{experiment:22s} {s['folds_completed']:>5d} "
            f"{format_value(s.get('mean_pck')):>8s} "
            f"{format_value(s.get('worst_pck')):>9s} "
            f"{str(s.get('worst_subject')):>10s} "
            f"{format_value(s.get('mean_sitting_pck')):>7s} "
            f"{format_value(s.get('mean_standing_pck')):>9s} "
            f"{format_value(s.get('mean_mpjpe'), '{:.4f}'):>7s} "
            f"{format_value(s.get('mean_classification_accuracy')):>7s}"
        )

    print("\nPER-FOLD PCK")
    subject_header = "".join(f"{s:>8s}" for s in ALL_SUBJECTS)
    print(f"{'experiment':22s}{subject_header}")
    for experiment, s in all_summaries.items():
        row = "".join(
            format_value(
                s["per_fold"].get(subject, {}).get("pck")
            ).rjust(8)
            for subject in ALL_SUBJECTS
        )
        print(f"{experiment:22s}{row}")

    suffix = "_adaptnorm" if args.adapt_norm else ""
    out_path = config.EVALUATION_DIR / f"loso_summary{suffix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

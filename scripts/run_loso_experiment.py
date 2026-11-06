"""
Run one experiment configuration under full leave-one-subject-out.

Unlike run_all_subject_tests.py, this never rewrites src/config.py: each
fold gets a JSON overrides file referenced through the CSI_CONFIG_OVERRIDES
environment variable (see the override hook at the bottom of src/config.py).

Usage:
    python scripts/run_loso_experiment.py --experiment rootrel_geo
    python scripts/run_loso_experiment.py --experiment baseline3c --subjects 001 002
    python scripts/run_loso_experiment.py --experiment rootrel_geo --epochs 2  # smoke
    python scripts/run_loso_experiment.py --experiment rootrel_geo_aug --adapt-norm-eval
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

ALL_SUBJECTS = ["001", "002", "003", "004", "005", "006", "007"]

# Experiment stages from the prioritized plan. Later stages include the
# earlier ones so the LOSO tables read as a cumulative ablation.
EXPERIMENTS = {
    # Baseline re-run on the current 3-class dataset (comparison anchor).
    "baseline3c": {},
    # P1: root-relative + scale-decomposed representation.
    "rootrel": {
        "POSE_REPRESENTATION": "root_relative",
    },
    # P2: + bone-length and joint-angle losses.
    "rootrel_geo": {
        "POSE_REPRESENTATION": "root_relative",
        "LAMBDA_BONE": 0.1,
        "LAMBDA_ANGLE": 0.1,
    },
    # P2 ablation without P1 (geometry loss on the absolute baseline).
    "geo_only": {
        "LAMBDA_BONE": 0.1,
        "LAMBDA_ANGLE": 0.1,
    },
    # P3: + CSI augmentation.
    "rootrel_geo_aug": {
        "POSE_REPRESENTATION": "root_relative",
        "LAMBDA_BONE": 0.1,
        "LAMBDA_ANGLE": 0.1,
        "AUGMENT_CSI": True,
    },
    # P4: + subject-adversarial feature learning.
    "rootrel_geo_aug_adv": {
        "POSE_REPRESENTATION": "root_relative",
        "LAMBDA_BONE": 0.1,
        "LAMBDA_ANGLE": 0.1,
        "AUGMENT_CSI": True,
        "ADVERSARIAL_SUBJECT": True,
    },
    # P5: GCN decoder (architecture counterpart of the geometry loss).
    "rootrel_gcn_aug": {
        "POSE_REPRESENTATION": "root_relative",
        "AUGMENT_CSI": True,
        "SAM_DECODER": "gcn",
    },
}


def run(command, env):
    print("\n" + "=" * 70)
    print("RUNNING:", " ".join(command))
    print("=" * 70, flush=True)
    subprocess.run(command, check=True, env=env, cwd=PROJECT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENTS),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=ALL_SUBJECTS,
        help="Subset of test subjects (default: all seven folds).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS (for smoke tests).",
    )
    parser.add_argument(
        "--adapt-norm-eval",
        action="store_true",
        help="Additionally evaluate each fold with AdaBN-style test-time "
        "normalization adaptation.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Evaluate existing checkpoints only.",
    )
    args = parser.parse_args()

    overrides_base = dict(EXPERIMENTS[args.experiment])
    if args.epochs is not None:
        overrides_base["NUM_EPOCHS"] = args.epochs

    python = sys.executable

    for test_id in args.subjects:
        remaining = [s for s in ALL_SUBJECTS if s != test_id]
        val_id = remaining[-1]

        tag = f"{args.experiment}_test{test_id}"

        overrides = dict(overrides_base)
        overrides["FIXED_TEST_SUBJECTS"] = [test_id]
        overrides["FIXED_VAL_SUBJECTS"] = [val_id]
        overrides["EXPERIMENT_TAG"] = tag

        with tempfile.NamedTemporaryFile(
            "w",
            suffix=f"_{tag}.json",
            delete=False,
        ) as f:
            json.dump(overrides, f, indent=2)
            overrides_path = f.name

        env = dict(os.environ)
        env["CSI_CONFIG_OVERRIDES"] = overrides_path
        env["PYTHONPATH"] = str(PROJECT)

        print("\n" + "#" * 80)
        print(f"EXPERIMENT {args.experiment} | TEST SUBJECT {test_id} | VAL {val_id}")
        print(f"overrides: {overrides_path}")
        print("#" * 80, flush=True)

        try:
            if not args.skip_train:
                run([python, "-m", "src.train"], env)
            run([python, "-m", "src.evaluate"], env)
            run([python, "-m", "src.evaluate_by_pose", "--split", "test"], env)
            if args.adapt_norm_eval:
                run([python, "-m", "src.evaluate", "--adapt-norm"], env)
        finally:
            os.unlink(overrides_path)

    print("\n" + "#" * 80)
    print(f"EXPERIMENT {args.experiment} FINISHED for subjects {args.subjects}")
    print("#" * 80)


if __name__ == "__main__":
    main()

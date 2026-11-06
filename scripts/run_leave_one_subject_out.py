import os
import subprocess
import json
from pathlib import Path


PROJECT = Path.cwd()

SUBJECTS = [
    "001",
]


def update_config(test_id):

    config_file = PROJECT / "src" / "config.py"

    text = config_file.read_text()

    # Test subject
    text = text.replace(
        'FIXED_TEST_SUBJECTS = ("007",)',
        f'FIXED_TEST_SUBJECTS = ("{test_id}",)'
    )

    # Use another subject for validation
    ALL_SUBJECTS = [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
    ]

    remaining = [
        s for s in ALL_SUBJECTS
        if s != test_id
    ]

    val_id = remaining[-1]

    text = text.replace(
        'FIXED_VAL_SUBJECTS = ("006",)',
        f'FIXED_VAL_SUBJECTS = ("{val_id}",)'
    )

    # Unique experiment name
    old = 'EXPERIMENT_TAG = "multipos_test007_localnorm"'

    new = (
        f'EXPERIMENT_TAG = '
        f'"leave_one_out_test{test_id}_localnorm"'
    )

    text = text.replace(
        old,
        new
    )

    config_file.write_text(text)

    print(
        f"\nCONFIGURED:"
        f"\n Test subject: {test_id}"
        f"\n Validation: {val_id}"
    )


def run(cmd):

    print("\nRUNNING:")
    print(cmd)

    subprocess.run(
        cmd,
        shell=True,
        check=True
    )


for test_id in SUBJECTS:

    print("\n" + "="*70)
    print(
        f"START TEST SUBJECT {test_id}"
    )
    print("="*70)


    update_config(test_id)


    # Train
    run(
        "python -m src.train"
    )


    # Evaluate
    run(
        "python -m src.evaluate"
    )


    # Pose breakdown
    run(
        "python -m src.evaluate_by_pose --split test"
    )


    # Visualization
    run(
        "python -m src.visualize "
        "--pose sitting "
        "--num-samples 10 "
        "--background original"
    )

    run(
        "python -m src.visualize "
        "--pose standing "
        "--num-samples 10 "
        "--background original"
    )


print("\nALL SUBJECT TESTS FINISHED")

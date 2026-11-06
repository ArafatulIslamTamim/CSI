import subprocess
from pathlib import Path


PROJECT = Path.cwd()


TEST_SUBJECTS = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
]


ALL_SUBJECTS = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
]


def run(cmd):

    print("\n" + "=" * 70)
    print("RUNNING:")
    print(cmd)
    print("=" * 70)

    subprocess.run(
        cmd,
        shell=True,
        check=True
    )


def update_config(test_id):

    config_file = PROJECT / "src" / "config.py"

    text = config_file.read_text()


    # Test subject
    text = replace_line(
        text,
        "FIXED_TEST_SUBJECTS",
        f'FIXED_TEST_SUBJECTS = ("{test_id}",)'
    )


    # Choose validation subject
    remaining = [
        s for s in ALL_SUBJECTS
        if s != test_id
    ]

    val_id = remaining[-1]


    text = replace_line(
        text,
        "FIXED_VAL_SUBJECTS",
        f'FIXED_VAL_SUBJECTS = ("{val_id}",)'
    )


    # Unique experiment name
    text = replace_line(
        text,
        "EXPERIMENT_TAG",
        f'EXPERIMENT_TAG = "leave_one_out_test{test_id}_localnorm"'
    )


    config_file.write_text(text)


    print(
        f"""
CONFIGURATION
--------------
Test subject : {test_id}
Validation   : {val_id}
Experiment   : leave_one_out_test{test_id}_localnorm
"""
    )


def replace_line(text, key, new_line):

    lines = text.splitlines()

    output = []

    for line in lines:

        if line.strip().startswith(key):

            output.append(new_line)

        else:

            output.append(line)

    return "\n".join(output)



for test_id in TEST_SUBJECTS:

    print("\n")
    print("#" * 80)
    print(f"STARTING TEST SUBJECT {test_id}")
    print("#" * 80)


    update_config(test_id)


    # Train
    run(
        "python -m src.train"
    )


    # Overall evaluation
    run(
        "python -m src.evaluate"
    )


    # Pose evaluation
    run(
        "python -m src.evaluate_by_pose --split test"
    )


    # Visualizations
    run(
        "python -m src.visualize "
        "--pose sitting "
        "--num-samples 20 "
        "--background original"
    )


    run(
        "python -m src.visualize "
        "--pose standing "
        "--num-samples 20 "
        "--background original"
    )


print("\n")
print("#" * 80)
print("ALL SUBJECT TESTS FINISHED")
print("#" * 80)

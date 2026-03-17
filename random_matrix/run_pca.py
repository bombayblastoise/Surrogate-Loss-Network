import os
import subprocess

SCRIPT = "./xgb_lib_pca.py"
N_JOBS = "32"

SKIP_FILES = {
    "application_data.csv",
    "card_transdata.csv",
}

FOLDERS = [
    "datasets",
    "large_datasets",
]

def run_experiments():
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        csv_files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".csv") and f not in SKIP_FILES
        )

        for csv_file in csv_files:
            dataset_path = os.path.join(folder, csv_file)

            cmd = [
                "python3",
                SCRIPT,
                "--n-jobs", N_JOBS,
                "--dataset-file", dataset_path,
            ]

            print(f"\nRunning on: {dataset_path}")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_experiments()

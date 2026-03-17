import json
from pathlib import Path
from collections import defaultdict
import logging
import shutil
import sys

# ---------------- CONFIG ----------------

INPUT_DIR = Path("jl_fl_results")          # directory with per-task JSONs
OUTPUT_DIR = Path("jl_fl_results_merged")  # final aggregated output
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ResultMerger")

# ----------------------------------------

def main():
    if not INPUT_DIR.exists():
        logger.error(f"Input directory {INPUT_DIR} does not exist.")
        sys.exit(1)

    files = list(INPUT_DIR.glob("*.json"))
    if not files:
        logger.error("No JSON files found in input directory. Aborting cleanup.")
        sys.exit(1)

    logger.info(f"Found {len(files)} JSON result files")

    datasets = defaultdict(lambda: {
        "baseline": None,
        "experiments": []
    })

    skipped = 0

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)

            dataset = data.get("dataset")
            task_type = data.get("type")

            if not dataset or not task_type:
                logger.warning(f"Skipping malformed file: {fp.name}")
                skipped += 1
                continue

            if task_type == "baseline":
                datasets[dataset]["baseline"] = {
                    "frozen": data["frozen"],
                    "tuned": data["tuned"],
                    "best_params": data["params"],
                }

            elif task_type == "experiment":
                datasets[dataset]["experiments"].append({
                    "split_method": data["split"],
                    "target_dim": data["dim"],
                    "frozen": data["frozen"],
                    "tuned": data["tuned"],
                    "best_params": data["params"],
                })

            else:
                logger.warning(f"Unknown task type in {fp.name}")
                skipped += 1

        except Exception as e:
            logger.error(f"Failed to read {fp.name}: {e}")
            skipped += 1

    # Write aggregated results
    written = 0
    for dataset, payload in datasets.items():
        out_path = OUTPUT_DIR / f"{dataset}_pytorch_results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        written += 1

    if written == 0:
        logger.error("No aggregated files written. Input directory will NOT be deleted.")
        sys.exit(1)

    logger.info(f"Aggregation complete")
    logger.info(f"Datasets written : {written}")
    logger.info(f"Files skipped    : {skipped}")
    logger.info(f"Output directory : {OUTPUT_DIR.resolve()}")

    # ---------------- CLEANUP ----------------
    logger.info(f"Removing input directory: {INPUT_DIR.resolve()}")
    shutil.rmtree(INPUT_DIR)
    logger.info("Cleanup complete. Raw per-task JSONs removed.")

if __name__ == "__main__":
    main()

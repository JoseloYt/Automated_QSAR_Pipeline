"""
run_pipeline.py

Master orchestrator for the automated QSAR pipeline.

Runs the complete pipeline:

    Step 1 - Fetch ChEMBL data
    Step 2 - Preprocess
    Step 3 - EDA & Lipinski
    Step 4 - Random Forest
    Step 5 - Chemprop (optional)

Example
-------
python run_pipeline.py --receptor FGFR1

python run_pipeline.py \
    --receptor FGFR1 \
    --bioactivity_type IC50

python run_pipeline.py \
    --receptor EGFR \
    --skip_chemprop
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import config

#Heavy imports so I'll make them lazy importing them in the main function
#import step1_fetch_chembl
#import step2_preprocess
#import step3_eda_lipinski
#import step4_rf_model
#import step5_chemprop_model 
import importlib


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logger(log_file: str) -> logging.Logger:
    """
    Configure console + file logging.
    """

    logger = logging.getLogger("QSARPipeline")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if executed multiple times
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------
# Pipeline step execution
# ---------------------------------------------------------------------
def execute_step(
    logger: logging.Logger,
    step_name: str,
    func,
    receptor: str,
    bioactivity_type: str | None,
):
    """
    Execute a pipeline step safely.

    Returns
    -------
    dict
        {
            success,
            elapsed,
            output,
            error
        }
    """

    logger.info("=" * 70)
    logger.info(step_name)

    start = time.perf_counter()

    try:

        output = func(receptor, bioactivity_type)

        elapsed = time.perf_counter() - start

        logger.info("✓ Completed in %.2f seconds", elapsed)

        if output:
            logger.info("Output: %s", output)

        return {
            "success": True,
            "elapsed": elapsed,
            "output": output,
            "error": None,
        }

    except Exception as e:

        elapsed = time.perf_counter() - start

        logger.exception("Step failed.")

        return {
            "success": False,
            "elapsed": elapsed,
            "output": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Automated QSAR Pipeline"
    )

    parser.add_argument(
        "--receptor",
        required=True,
        help="Target receptor name (e.g. FGFR1)"
    )

    parser.add_argument(
        "--bioactivity_type",
        default=None,
        help=f"Bioactivity type (default: {config.BIOACTIVITY_TYPE})"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory (currently informational only)."
    )

    parser.add_argument(
        "--skip_chemprop",
        action="store_true",
        help="Skip Chemprop model training."
    )

    args = parser.parse_args()

    receptor = args.receptor
    bioactivity = args.bioactivity_type

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    output_dir = config.get_output_dir(receptor)

    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "pipeline.log")

    logger = setup_logger(log_file)

    logger.info("")
    logger.info("=" * 70)
    logger.info("AUTOMATED QSAR PIPELINE")
    logger.info("=" * 70)
    logger.info("Receptor         : %s", receptor)
    logger.info(
        "Bioactivity type : %s",
        bioactivity if bioactivity else config.BIOACTIVITY_TYPE,
    )
    logger.info("Output directory : %s", output_dir)
    logger.info("=" * 70)

    total_start = time.perf_counter()

    results = {}

    pipeline_steps = [
        (
            "Step 1 - Fetch ChEMBL Data",
            "step1_fetch_chembl",
        ),
        (
            "Step 2 - Preprocessing",
            "step2_preprocess",
        ),
        (
            "Step 3 - EDA & Lipinski",
            "step3_eda_lipinski",
        ),
        (
            "Step 4 - Random Forest",
            "step4_rf_model",
        ),
    ]

    if not args.skip_chemprop:
        pipeline_steps.append(
            (
                "Step 5 - Chemprop",
                "step5_chemprop_model",
            )
        )

    # ---------------------------------------------------------
    # Execute pipeline
    # ---------------------------------------------------------

    for name, module_name in pipeline_steps:

        mod = importlib.import_module(module_name)

        results[name] = execute_step(
            logger,
            name,
            mod.run,
            receptor,
            bioactivity,
        )

    total_elapsed = time.perf_counter() - total_start

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)

    for step, result in results.items():

        status = "SUCCESS" if result["success"] else "FAILED"

        logger.info(
            "%-30s %-8s %.2f s",
            step,
            status,
            result["elapsed"],
        )

        if not result["success"]:
            logger.info("    Error : %s", result["error"])

    logger.info("=" * 70)
    logger.info("Total execution time : %.2f seconds", total_elapsed)
    logger.info("Log saved to         : %s", log_file)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
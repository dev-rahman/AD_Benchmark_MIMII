"""
run_all.py — Run all 6 models on all 3 machine types and produce a
             unified comparison table for the paper.

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python run_all.py [--machine fan|pump|valve|all] [--models all|classical|dl]
"""

import argparse
import importlib
import os
import sys
import time

import pandas as pd

from utils.config import MACHINES, RESULTS_DIR

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    # name           : (module path, category)
    "ocsvm":          ("Baseline.SVM.ocsvm",                       "Classical ML"),
    "iforest":        ("Baseline.IsolationForest.iforest",          "Classical ML"),
    "dense_ae":       ("Autoencoder.Dense.dense_ae",               "DL – Reconstruction"),
    "cnn_ae":         ("Autoencoder.CNN.cnn_ae",                   "DL – Reconstruction"),
    "transformer_ae": ("Autoencoder.Transformer.transformer_ae",   "DL – Reconstruction"),
    "cnn14_knn":      ("Embedding.CNN14_KNN.cnn14_knn",            "Hybrid (DL Emb.)"),
}

CLASSICAL = {"ocsvm", "iforest"}
DL        = {"dense_ae", "cnn_ae", "transformer_ae", "cnn14_knn"}


def run_model(name: str, machine: str) -> dict | None:
    module_path, category = MODELS[name]
    try:
        mod    = importlib.import_module(module_path)
        result = mod.run(machine)
        result["category"] = category
        return result
    except Exception as e:
        print(f"  ⚠ {name} | {machine} failed: {e}")
        return None


def print_table(df: pd.DataFrame) -> None:
    cols = ["method", "category", "machine", "AUC", "pAUC", "F1"]
    print("
" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df[cols].to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run full MIMII benchmark")
    parser.add_argument("--machine", default="all",
                        choices=MACHINES + ["all"])
    parser.add_argument("--models",  default="all",
                        choices=["all", "classical", "dl"])
    args = parser.parse_args()

    machines      = MACHINES if args.machine == "all" else [args.machine]
    model_names   = list(MODELS.keys())
    if args.models == "classical":
        model_names = [m for m in model_names if m in CLASSICAL]
    elif args.models == "dl":
        model_names = [m for m in model_names if m in DL]

    print(f"\nRunning {len(model_names)} models × {len(machines)} machines …\n")

    all_results = []
    t0 = time.time()

    for name in model_names:
        for machine in machines:
            result = run_model(name, machine)
            if result:
                all_results.append(result)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Full results → {out_path}")
    print_table(df)

    # Average AUC per method across machines
    print("\n── Average AUC per method ──")
    avg = (
        df.groupby(["method", "category"])[["AUC", "pAUC", "F1"]]
        .mean()
        .round(4)
        .sort_values("AUC", ascending=False)
    )
    print(avg.to_string())
    avg_path = os.path.join(RESULTS_DIR, "benchmark_avg.csv")
    avg.reset_index().to_csv(avg_path, index=False)
    print(f"Average results → {avg_path}")


if __name__ == "__main__":
    main()

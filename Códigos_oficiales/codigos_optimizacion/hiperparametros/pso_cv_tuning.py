import os
import sys
import json
import time
import traceback
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import current_process

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation
from pymoo_opt import (
    MODEL_2264,
    PSO_CONFIG,
    run_pymoo_estimation,
    prepare_model_structure,
    objective_function_multi,
    objective_function,
)


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes
    return f"{minutes} min {rem_seconds:.2f} s" if minutes else f"{rem_seconds:.2f} s"


def format_elapsed_hms(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours} h {minutes} min {secs} s"


def log_fold_completion(combo_id, fold_id, elapsed_seconds):
    worker_name = current_process().name
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{timestamp}] worker={worker_name} | combo_id={combo_id} | fold_id={fold_id} | "
        f"duracion={format_elapsed_hms(elapsed_seconds)}"
    )


def build_datasets(paths, t_muestreo=3.0):
    datasets = []

    for i, path in enumerate(paths):
        data_excel = data_for_simulation(path, t_muestreo=t_muestreo)

        dataset = {
            "dataset_id": i,
            "path": path,
            "name": os.path.splitext(os.path.basename(path))[0],
            "x0": data_excel[0],
            "t_rel": data_excel[1],
            "sugars_profile": data_excel[2],
            "temp": data_excel[3],
            "Nadd": data_excel[4],
            "t_span": data_excel[5],
            "Et_final_exp": data_excel[6],
        }
        datasets.append(dataset)

    return datasets


def build_cv_folds(datasets):
    folds = []
    n = len(datasets)

    for val_idx in range(n):
        train_datasets = [datasets[i] for i in range(n) if i != val_idx]
        val_dataset = datasets[val_idx]

        folds.append(
            {
                "fold_id": val_idx + 1,
                "train_indices": [d["dataset_id"] for d in train_datasets],
                "val_index": val_dataset["dataset_id"],
                "train_datasets": train_datasets,
                "val_dataset": val_dataset,
            }
        )

    return folds


def build_hyperparameter_grid(
    c1_options,
    c2_options,
    w_options,
    pop_size_options,
):
    combos = []
    combo_id = 1

    for c1, c2, w, pop_size in product(
        c1_options,
        c2_options,
        w_options,
        pop_size_options,
    ):
        combos.append(
            {
                "combo_id": combo_id,
                "c1": float(c1),
                "c2": float(c2),
                "w": float(w),
                "pop_size": int(pop_size),
            }
        )
        combo_id += 1

    return combos


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def serialize_for_excel(value):
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False)
    return value


def _fold_record_to_dict(record):
    fold_columns = [
        "combo_id",
        "fold_id",
        "c1",
        "c2",
        "w",
        "pop_size",
        "train_indices",
        "val_index",
        "train_names",
        "train_cost_total",
        "val_cost",
        "elapsed_seconds",
        "best_theta_free",
    ]
    return {
        k: serialize_for_excel(record.get(k, np.nan))
        for k in fold_columns
    }


def _summary_record_to_dict(record):
    return {k: serialize_for_excel(v) for k, v in record.items()}


def evaluate_single_fold(
    fold,
    combo,
    model_structure,
    base_pso_config,
):
    fold_id = fold["fold_id"]
    train_datasets = fold["train_datasets"]
    val_dataset = fold["val_dataset"]

    pso_config = base_pso_config.copy()
    pso_config["c1"] = combo["c1"]
    pso_config["c2"] = combo["c2"]
    pso_config["w"] = combo["w"]
    pso_config["pop_size"] = combo["pop_size"]
    pso_config["epoch"] = base_pso_config.get("epoch", PSO_CONFIG.get("epoch", 1000))
    pso_config["relative_gap_threshold"] = base_pso_config.get(
        "relative_gap_threshold",
        PSO_CONFIG.get("relative_gap_threshold", 0.05),
    )

    fixed_params, free_names, _ = prepare_model_structure(model_structure)

    t0 = time.perf_counter()

    try:
        result, best_params = run_pymoo_estimation(
            model_structure=model_structure,
            datasets=train_datasets,
            pso_config=pso_config,
        )
        best_theta = np.asarray(result["x"], dtype=float)

        train_cost_total = objective_function_multi(
            theta=best_theta,
            free_names=free_names,
            fixed_params=fixed_params,
            datasets=train_datasets,
        )
        train_cost_mean = float(train_cost_total) / len(train_datasets)

        val_cost = objective_function(
            theta=best_theta,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=val_dataset["x0"],
            t_rel=val_dataset["t_rel"],
            temp=val_dataset["temp"],
            Nadd=val_dataset["Nadd"],
            t_span=val_dataset["t_span"],
            sugars_profile=val_dataset["sugars_profile"],
            Et_final_exp=val_dataset["Et_final_exp"],
        )

        elapsed_seconds = time.perf_counter() - t0
        log_fold_completion(combo["combo_id"], fold_id, elapsed_seconds)

        return {
            "combo_id": combo["combo_id"],
            "fold_id": fold_id,
            "c1": combo["c1"],
            "c2": combo["c2"],
            "w": combo["w"],
            "pop_size": combo["pop_size"],
            "train_indices": fold["train_indices"],
            "val_index": fold["val_index"],
            "train_names": [d["name"] for d in train_datasets],
            "train_cost_total": float(train_cost_total),
            "val_cost": float(val_cost),
            "elapsed_seconds": float(elapsed_seconds),
            "best_theta_free": best_theta.tolist(),
        }

    except Exception as e:
        elapsed_seconds = time.perf_counter() - t0
        log_fold_completion(combo["combo_id"], fold_id, elapsed_seconds)

        return {
            "combo_id": combo["combo_id"],
            "fold_id": fold_id,
            "c1": combo["c1"],
            "c2": combo["c2"],
            "w": combo["w"],
            "pop_size": combo["pop_size"],
            "train_indices": fold["train_indices"],
            "val_index": fold["val_index"],
            "train_names": [d["name"] for d in train_datasets],
            "train_cost_total": np.nan,
            "val_cost": np.nan,
            "elapsed_seconds": float(elapsed_seconds),
            "best_theta_free": [],
            "_status": "failed",
            "_error_message": f"{type(e).__name__}: {str(e)}",
            "_traceback": traceback.format_exc(),
        }


def summarize_combo_records(combo, fold_records):
    ok_records = [r for r in fold_records if r.get("_status", "ok") == "ok"]
    failed_records = [r for r in fold_records if r.get("_status", "ok") != "ok"]

    train_costs = [r["train_cost_total"] for r in ok_records]
    val_costs = [r["val_cost"] for r in ok_records]

    return {
        "combo_id": combo["combo_id"],
        "c1": combo["c1"],
        "c2": combo["c2"],
        "w": combo["w"],
        "pop_size": combo["pop_size"],
        "n_folds_failed": len(failed_records),
        "mean_train_cost": float(np.nanmean(train_costs)) if train_costs else np.nan,
        "std_train_cost": float(np.nanstd(train_costs)) if train_costs else np.nan,
        "mean_val_cost": float(np.nanmean(val_costs)) if val_costs else np.nan,
        "std_val_cost": float(np.nanstd(val_costs)) if val_costs else np.nan,
        "min_val_cost": float(np.nanmin(val_costs)) if val_costs else np.nan,
        "max_val_cost": float(np.nanmax(val_costs)) if val_costs else np.nan,
    }


def evaluate_hyperparameter_combo(
    combo,
    folds,
    model_structure,
    base_pso_config,
):
    fold_records = []

    for fold in folds:
        record = evaluate_single_fold(
            fold=fold,
            combo=combo,
            model_structure=model_structure,
            base_pso_config=base_pso_config,
        )
        fold_records.append(record)

    summary = summarize_combo_records(combo, fold_records)
    return {
        "combo": combo,
        "fold_records": fold_records,
        "summary": summary,
    }


def _combo_worker(args):
    combo, folds, model_structure, base_pso_config = args
    return evaluate_hyperparameter_combo(
        combo=combo,
        folds=folds,
        model_structure=model_structure,
        base_pso_config=base_pso_config,
    )


def run_cv_hyperparameter_search(
    paths,
    c1_options,
    c2_options,
    w_options,
    pop_size_options,
    output_dir,
    model_structure=None,
    base_pso_config=None,
    t_muestreo=3.0,
    n_workers=4,
    parallel=True,
):
    if model_structure is None:
        model_structure = MODEL_2264

    if base_pso_config is None:
        base_pso_config = PSO_CONFIG.copy()
    else:
        tmp = PSO_CONFIG.copy()
        tmp.update(base_pso_config)
        base_pso_config = tmp

    base_pso_config["verbose"] = False
    base_pso_config["save_history"] = False

    ensure_dir(output_dir)

    datasets = build_datasets(paths, t_muestreo=t_muestreo)
    folds = build_cv_folds(datasets)

    combos = build_hyperparameter_grid(
        c1_options=c1_options,
        c2_options=c2_options,
        w_options=w_options,
        pop_size_options=pop_size_options,
    )

    fold_xlsx = os.path.join(output_dir, "cv_fold_results.xlsx")
    combo_xlsx = os.path.join(output_dir, "cv_combo_summary.xlsx")
    best_json = os.path.join(output_dir, "best_result.json")

    all_fold_records = []
    all_combo_records = []
    all_combo_summaries = []
    t_global_0 = time.perf_counter()

    if parallel:
        max_workers = max(1, int(n_workers))

        args_list = [
            (combo, folds, model_structure, base_pso_config)
            for combo in combos
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_combo_worker, args) for args in args_list]

            for future in as_completed(futures):
                result = future.result()

                fold_rows = [_fold_record_to_dict(r) for r in result["fold_records"]]
                summary_row = _summary_record_to_dict(result["summary"])

                all_fold_records.extend(fold_rows)
                all_combo_records.append(summary_row)
                all_combo_summaries.append(result["summary"])
                
                print(f"[{len(all_combo_summaries)}/{len(combos)}] "
                    f"Combinación terminada | combo_id={result['summary']['combo_id']} | "
                    f"val_mean={result['summary']['mean_val_cost']:.6f} | "
                    f"train_mean={result['summary']['mean_train_cost']:.6f}"
                )
    else:
        for combo in combos:
            result = evaluate_hyperparameter_combo(
                combo=combo,
                folds=folds,
                model_structure=model_structure,
                base_pso_config=base_pso_config,
            )

            fold_rows = [_fold_record_to_dict(r) for r in result["fold_records"]]
            summary_row = _summary_record_to_dict(result["summary"])

            all_fold_records.extend(fold_rows)
            all_combo_records.append(summary_row)
            all_combo_summaries.append(result["summary"])

    total_elapsed = time.perf_counter() - t_global_0

    # Guardar fold_records en Excel
    if all_fold_records:
        df_fold = pd.DataFrame(all_fold_records)
        df_fold.to_excel(fold_xlsx, index=False, engine='openpyxl')
    
    # Guardar combo_records en Excel
    if all_combo_records:
        df_combo = pd.DataFrame(all_combo_records)
        df_combo.to_excel(combo_xlsx, index=False, engine='openpyxl')

    df_summary = pd.DataFrame(all_combo_summaries)
    
    best_row = df_summary.loc[df_summary["mean_val_cost"].idxmin()].to_dict() if not df_summary.empty else {}

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_result": best_row,
                "total_elapsed_seconds": total_elapsed,
                "total_elapsed_human": format_elapsed(total_elapsed),
                "n_datasets": len(datasets),
                "n_folds": len(folds),
                "n_combinations": len(combos),
                "n_total_pso_runs": len(combos) * len(folds),
                "output_files": {
                    "fold_results": fold_xlsx,
                    "combo_summary": combo_xlsx,
                    "best_result_json": best_json,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "datasets": datasets,
        "folds": folds,
        "combos": combos,
        "summary_df": df_summary,
        "best_result": best_row,
        "output_dir": output_dir,
        "elapsed_seconds": total_elapsed,
        "best_combo_id": int(best_row.get("combo_id", -1)),
        "best_mean_val_cost": float(best_row.get("mean_val_cost", float('nan'))),
    }
# -*- coding: utf-8 -*-
"""
Validación cruzada aleatoria repetida con pymoo PSO
- 20 datasets CS de 100.000 L
- Se separan 4 datasets una sola vez al inicio
- En cada iteración: 5 ajuste usando solo los 12 restantes
- Sin repetir combinaciones de ajuste dentro de la misma corrida
- Ejecución paralela con workers
- Guardado final a Excel
- Seed fija solo para PSO, no para los splits aleatorios
"""

import os
import sys
import time
import math
import random
import traceback
from itertools import combinations
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation, simulate_system
from pymoo_opt import (
    MODEL_2264,
    PARAM_ORDER,
    PSO_CONFIG,
    run_pymoo_estimation,
    params_dict_to_vector,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

N_ITERATIONS = 50
N_VALIDATION = 0
N_HOLDOUT = 4
N_TRAIN = 5
MAX_WORKERS = 5

OUTPUT_BASENAME = "resultados_cv_pymoo_100k"

MODEL_STRUCTURE = MODEL_2264

CUSTOM_PSO_CONFIG = PSO_CONFIG.copy()
CUSTOM_PSO_CONFIG["epoch"] = 2000
CUSTOM_PSO_CONFIG["pop_size"] = 25
CUSTOM_PSO_CONFIG["w"] = 0.5
CUSTOM_PSO_CONFIG["c1"] = 1.5
CUSTOM_PSO_CONFIG["c2"] = 1.5
CUSTOM_PSO_CONFIG["seed"] = 123
CUSTOM_PSO_CONFIG["verbose"] = False
CUSTOM_PSO_CONFIG["save_history"] = False
CUSTOM_PSO_CONFIG["relative_gap_threshold"] = 1e-4 # antes estaba en 1e-3


# ============================================================
# DATASETS 100.000 L
# ============================================================

DATASETS_INFO = [
    {
        "id": 1,
        "name": "Data CS 24 EL BOLDO estanque 30.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    },
    {
        "id": 2,
        "name": "Data CS 24 LOU estanque 54.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    },
    {
        "id": 3,
        "name": "Data CS 25 EL BOLDO estanque 55.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    },
    {
        "id": 4,
        "name": "Data CS 25 LOU estanque 61.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",
    },
    {
        "id": 5,
        "name": "Data SY 24 LOU+VAL+FN estanque 36.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU+VAL+FN estanque 36.xlsx",
    },
    {
        "id": 6,
        "name": "Data SY 24 VAL+STARAQ estanque 56.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 VAL+STARAQ estanque 56.xlsx",
    },
    {
        "id": 7,
        "name": "Data SY 24 LOU estanque 62.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU estanque 62.xlsx",
    },
    {
        "id": 8,
        "name": "Data SY 25 LOU estanque 30.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 25 LOU estanque 30.xlsx",
    },
    {
        "id": 9,
        "name": "Data ME 25 Q. AGUA estanque 85.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 Q. AGUA estanque 85.xlsx",
    },
    {
        "id": 10,
        "name": "Data ME 24 QAGUA estanque 54.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 24 QAGUA estanque 54.xlsx",
    },
    {
        "id": 11,
        "name": "Data ME 25 AURORA + STA MARTA estanque 57.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 AURORA + STA MARTA estanque 57.xlsx",
    },
    {
        "id": 12,
        "name": "Data ME 25 STA MARTA estanque 62.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 STA MARTA estanque 62.xlsx",
    },
    {
        "id": 13,
        "name": "Data CA 24 VAL estanque 31.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 31.xlsx",
    },
    {
        "id": 14,
        "name": "Data CA 24 VAL estanque 59.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 59.xlsx",
    },
    {
        "id": 15,
        "name": "Data CA 24 VAL estanque 62.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 62.xlsx",
    },
    {
        "id": 16,
        "name": "Data CA 25 F.N. estanque 68.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 25 F.N. estanque 68.xlsx",
    },
]


# ============================================================
# UTILIDADES
# ============================================================

def format_elapsed(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes
    return f"{minutes} min {rem_seconds:.2f} s" if minutes else f"{rem_seconds:.2f} s"


def compute_validation_cost(sol, sugars_profile, Et_final_exp, penalty=1e12, eps=1e-8):
    y = sol.y.T
    sugars_sim = np.asarray(y[:, 2] + y[:, 3], dtype=float)
    Et_final_sim = float(y[-1, 4])

    sugars_profile = np.asarray(sugars_profile, dtype=float)
    Et_final_exp = float(Et_final_exp)

    if len(sugars_sim) != len(sugars_profile):
        return {
            "sugar_error_mean": penalty,
            "ethanol_error": penalty,
            "objective_total": penalty,
            "Et_final_sim": Et_final_sim,
        }

    if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
        return {
            "sugar_error_mean": penalty,
            "ethanol_error": penalty,
            "objective_total": penalty,
            "Et_final_sim": Et_final_sim,
        }

    sugar_scale = max(np.max(np.abs(sugars_profile)), eps)
    ethanol_scale = max(abs(Et_final_exp), eps)

    sugar_res = (sugars_sim - sugars_profile) / sugar_scale
    etoh_res = (Et_final_sim - Et_final_exp) / ethanol_scale

    sugar_error_mean = float(np.mean(sugar_res ** 2))
    ethanol_error = float(etoh_res ** 2)
    objective_total = float(sugar_error_mean + ethanol_error)

    return {
        "sugar_error_mean": sugar_error_mean,
        "ethanol_error": ethanol_error,
        "objective_total": objective_total,
        "Et_final_sim": Et_final_sim,
    }


def build_all_datasets(datasets_info):
    datasets_by_id = {}

    print("\nConstruyendo datasets...")
    for item in datasets_info:
        path = item["path"]
        data_excel = data_for_simulation(path)

        datasets_by_id[item["id"]] = {
            "id": item["id"],
            "name": item["name"],
            "path": path,
            "x0": data_excel[0],
            "t_rel": data_excel[1],
            "sugars_profile": data_excel[2],
            "temp": data_excel[3],
            "Nadd": data_excel[4],
            "t_span": data_excel[5],
            "Et_final_exp": data_excel[6],
        }

        print(f"[OK] Dataset {item['id']:02d}: {item['name']}")

    return datasets_by_id


def generate_unique_validation_splits(dataset_ids, n_train, n_iterations, holdout_ids):
    remaining_ids = sorted(set(dataset_ids) - set(holdout_ids))
    all_combos = list(combinations(remaining_ids, n_train))

    max_possible = len(all_combos)
    if n_iterations > max_possible:
        raise ValueError(
            f"Se pidieron {n_iterations} iteraciones, pero solo existen "
            f"{max_possible} combinaciones únicas de ajuste con los datasets restantes."
        )

    selected = random.sample(all_combos, n_iterations)

    splits = []
    holdout_ids = tuple(sorted(holdout_ids))

    for i, train_ids in enumerate(selected, start=1):
        train_ids = tuple(sorted(train_ids))

        splits.append({
            "iteration": i,
            "train_ids": train_ids,
            "val_ids": holdout_ids,
        })

    return splits


def save_results_to_excel(results, output_excel):
    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="iteracion").reset_index(drop=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="resultados", index=False)


def create_output_dir():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CURRENT_DIR, f"{OUTPUT_BASENAME}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================
# WORKER
# ============================================================

def run_single_split(split, datasets_by_id, model_structure, pso_config):
    start = time.perf_counter()

    iteration = split["iteration"]
    train_ids = split["train_ids"]
    separated_ids = split["val_ids"]

    try:
        train_datasets = [datasets_by_id[i] for i in train_ids]

        result_fit, best_params = run_pymoo_estimation(
            model_structure=model_structure,
            datasets=train_datasets,
            pso_config=pso_config
        )

        best_params_vector = params_dict_to_vector(best_params, PARAM_ORDER)

        costo_ajuste = float(result_fit["fun"])
        costo_validacion = 0.0
        costo_total = float(costo_ajuste)

        elapsed = time.perf_counter() - start
        end_clock = datetime.now()

        termination_info = result_fit.get("termination_info", {})

        row = {
            "iteracion": iteration,
            "hora_fin": end_clock.strftime("%Y-%m-%d %H:%M:%S"),
            "duracion_s": float(elapsed),
            "duracion_min": float(elapsed / 60.0),

            "ids_ajuste": ",".join(map(str, train_ids)),
            "ids_validacion": "",
            "ids_separados": ",".join(map(str, separated_ids)),

            "nombres_ajuste": " | ".join(datasets_by_id[i]["name"] for i in train_ids),
            "nombres_validacion": "",
            "nombres_separados": " | ".join(datasets_by_id[i]["name"] for i in separated_ids),

            "costo_ajuste": costo_ajuste,
            "costo_validacion": costo_validacion,
            "costo_total": costo_total,

            "stop_reason": termination_info.get("stop_reason"),
            "stop_iteration": termination_info.get("stop_iteration"),
            "final_ratio": termination_info.get("final_ratio"),
            "umbral_relativo": termination_info.get("threshold"),
            "max_epoch": termination_info.get("max_epoch"),
        }

        for param_name, param_value in zip(PARAM_ORDER, best_params_vector):
            row[param_name] = float(param_value)

        summary = {
            "iteracion": iteration,
            "costo_ajuste": costo_ajuste,
            "costo_validacion": costo_validacion,
            "costo_total": costo_total,
            "duracion_s": float(elapsed),
            "hora_fin": row["hora_fin"],
            "umbral_relativo": termination_info.get("threshold"),
            "stop_iteration": termination_info.get("stop_iteration"),
            "final_ratio": termination_info.get("final_ratio"),
        }

        return {
            "ok": True,
            "row": row,
            "summary": summary,
            "validation_details": [],
        }

    except Exception as e:
        elapsed = time.perf_counter() - start
        end_clock = datetime.now()

        row = {
            "iteracion": iteration,
            "hora_fin": end_clock.strftime("%Y-%m-%d %H:%M:%S"),
            "duracion_s": float(elapsed),
            "duracion_min": float(elapsed / 60.0),

            "ids_ajuste": ",".join(map(str, train_ids)),
            "ids_validacion": "",
            "ids_separados": ",".join(map(str, separated_ids)),

            "nombres_ajuste": " | ".join(datasets_by_id[i]["name"] for i in train_ids),
            "nombres_validacion": "",
            "nombres_separados": " | ".join(datasets_by_id[i]["name"] for i in separated_ids),

            "costo_ajuste": np.nan,
            "costo_validacion": np.nan,
            "costo_total": np.nan,

            "stop_reason": "error",
            "stop_iteration": np.nan,
            "final_ratio": np.nan,
            "umbral_relativo": np.nan,
            "max_epoch": np.nan,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }

        for param_name in PARAM_ORDER:
            row[param_name] = np.nan

        summary = {
            "iteracion": iteration,
            "costo_ajuste": np.nan,
            "costo_validacion": np.nan,
            "costo_total": np.nan,
            "duracion_s": float(elapsed),
            "hora_fin": row["hora_fin"],
            "umbral_relativo": np.nan,
            "stop_iteration": np.nan,
            "final_ratio": np.nan,
        }

        return {
            "ok": False,
            "row": row,
            "summary": summary,
            "validation_details": [],
        }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("VALIDACIÓN CRUZADA ALEATORIA REPETIDA - PYMOO PSO")
    print("=" * 80)

    if len(DATASETS_INFO) == 0:
        raise ValueError("DATASETS_INFO está vacío.")

    dataset_ids = [item["id"] for item in DATASETS_INFO]

    if N_TRAIN + N_HOLDOUT > len(dataset_ids):
        raise ValueError(
            "N_TRAIN + conjuntos separados no puede ser mayor al total de datasets."
        )

    holdout_ids = tuple(sorted(random.sample(dataset_ids, N_HOLDOUT)))
    remaining_ids = sorted(set(dataset_ids) - set(holdout_ids))
    total_possible = math.comb(len(remaining_ids), N_TRAIN)
    print(f"Sets separados fijos: {holdout_ids}")
    print(f"Cantidad total de combinaciones únicas posibles: {total_possible}")
    print(f"Iteraciones pedidas: {N_ITERATIONS}")
    print(f"Workers: {MAX_WORKERS}")
    output_dir = create_output_dir()
    output_excel = os.path.join(output_dir, f"{OUTPUT_BASENAME}.xlsx")
    output_excel_details = os.path.join(output_dir, f"{OUTPUT_BASENAME}_detalles_validacion.xlsx")
    print(f"Carpeta de salida: {output_dir}")
    print(f"Archivo principal: {output_excel}")
    print(f"Archivo de detalle: {output_excel_details}")
    print("Semilla PSO fija: 123")
    print("Splits aleatorios sin seed fija")

    datasets_by_id = build_all_datasets(DATASETS_INFO)

    splits = generate_unique_validation_splits(
        dataset_ids=dataset_ids,
        n_train=N_TRAIN,
        n_iterations=N_ITERATIONS,
        holdout_ids=holdout_ids,
    )

    print("\nSplits generados correctamente.")
    print("Ejemplos de separación:")
    for split in splits[:min(5, len(splits))]:
        print(
            f"Iteración {split['iteration']:02d} -> "
            f"separados {split['val_ids']} | ajuste {split['train_ids']}"
        )

    results = []
    validation_details_all = []
    global_start = time.perf_counter()

    print("\nIniciando ejecución paralela...\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                run_single_split,
                split,
                datasets_by_id,
                MODEL_STRUCTURE,
                CUSTOM_PSO_CONFIG
            ): split
            for split in splits
        }

        for future in as_completed(futures):
            split = futures[future]

            try:
                out = future.result()
                row = out["row"]
                summary = out["summary"]
                results.append(row)
                validation_details_all.extend(out.get("validation_details", []))

                status = "OK" if out["ok"] else "ERROR"

                print(
                    f"[{status}] Iteración {summary['iteracion']:03d} | "
                    f"Ajuste = {summary['costo_ajuste']:.6f} | "
                    f"Separados = {summary['costo_validacion']:.6f} | "
                    f"Total = {summary['costo_total']:.6f} | "
                    f"Duración = {format_elapsed(summary['duracion_s'])} | "
                    f"Hora = {summary['hora_fin']}"
                )

            except Exception as e:
                print(
                    f"[ERROR GRAVE] Falló future de la iteración "
                    f"{split['iteration']:03d}: {e}"
                )

    total_elapsed = time.perf_counter() - global_start

    save_results_to_excel(results, output_excel)

    if validation_details_all:
        df_details = pd.DataFrame(validation_details_all)
        df_details = df_details.sort_values(by=["iteracion", "id_validacion"]).reset_index(drop=True)
        with pd.ExcelWriter(output_excel_details, engine="openpyxl") as writer:
            df_details.to_excel(writer, sheet_name="detalles_validacion", index=False)

    print("\n" + "=" * 80)
    print("EJECUCIÓN FINALIZADA")
    print("=" * 80)
    print(f"Tiempo total: {format_elapsed(total_elapsed)}")
    print(f"Resultados guardados en: {output_excel}")

    if results:
        df = pd.DataFrame(results).sort_values(by="iteracion").reset_index(drop=True)

        print("\nResumen final:")
        print(f"Corridas registradas: {len(df)}")

        if "costo_total" in df.columns:
            valid_costs = df["costo_total"].dropna()
            if len(valid_costs) > 0:
                best_idx = valid_costs.idxmin()
                best_row = df.loc[best_idx]

                print(f"Mejor iteración: {int(best_row['iteracion'])}")
                print(f"Mejor costo total: {best_row['costo_total']:.6f}")
                print(f"Sets separados de esa iteración: {best_row['ids_separados']}")


if __name__ == "__main__":
    main()
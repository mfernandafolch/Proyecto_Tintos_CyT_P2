"""
Cálculo del costo de validación usando la función de costo original
del código de optimización.

Este script calcula el costo de la "línea negra", es decir:
- parámetros libres = mediana de los resultados obtenidos
- parámetros fijos = FIXED_PARAMS
- datasets = VALIDATION_DATASET_IDS

Importante:
NO redefine la función de costo.
Llama directamente a:
- objective_function_multi
- compute_objective_breakdown
desde pymoo_opt.py
"""

import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from simulacion import data_for_simulation

from pymoo_opt import (
    objective_function_multi,
    compute_objective_breakdown,
)


# ============================================================
# IMPORTAR CONFIGURACIÓN DESDE TU SCRIPT DE INCERTIDUMBRE
# ============================================================
# Este archivo debe estar en la misma carpeta que main_plot_uncertainty.py.
# Si tu archivo tiene otro nombre, cambia este import.

from main_plot_uncertainty import (
    DATASETS_INFO,
    VALIDATION_DATASET_IDS,
    FREE_PARAM_SAMPLES,
    FIXED_PARAMS,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PENALTY_COST = 1e12

# ============================================================
# UTILIDADES
# ============================================================

def choose_datasets_by_ids(datasets_info, dataset_ids):
    dataset_map = {item["id"]: item for item in datasets_info}

    missing = [
        dataset_id for dataset_id in dataset_ids
        if dataset_id not in dataset_map
    ]

    if missing:
        raise ValueError(f"IDs no encontrados en DATASETS_INFO: {missing}")

    return [dataset_map[dataset_id] for dataset_id in dataset_ids]


def build_dataset(item):
    """
    Construye un dataset con el mismo formato usado por objective_function_multi.
    """

    data_excel = data_for_simulation(item["path"])

    return {
        "id": item["id"],
        "name": item["name"],
        "path": item["path"],
        "x0": data_excel[0],
        "t_rel": data_excel[1],
        "sugars_profile": np.asarray(data_excel[2], dtype=float),
        "temp": data_excel[3],
        "Nadd": data_excel[4],
        "t_span": data_excel[5],
        "Et_final_exp": float(data_excel[6]),
    }


def build_validation_datasets():
    selected_info = choose_datasets_by_ids(
        DATASETS_INFO,
        VALIDATION_DATASET_IDS
    )

    datasets = []

    print("\nCargando datasets de validación:")

    for item in selected_info:
        print(f"  Dataset {item['id']:02d}: {item['name']}")
        datasets.append(build_dataset(item))

    return datasets


def build_theta_median_from_samples(free_param_samples):
    """
    Construye:
    - free_names: nombres de parámetros libres, en el mismo orden del diccionario
    - theta_median: vector de medianas en ese mismo orden

    Esto es clave porque objective_function_multi recibe:
        theta, free_names, fixed_params, datasets
    """

    print(f"\nCantidad de parámetros en free_param_samples: {len(FREE_PARAM_SAMPLES['mu0'])}")

    free_names = list(free_param_samples.keys())

    theta_median = np.array([
        np.nanmedian(np.asarray(free_param_samples[name], dtype=float))
        for name in free_names
    ], dtype=float)

    return free_names, theta_median


# ============================================================
# CÁLCULO DE COSTOS
# ============================================================

def compute_dataset_breakdowns(theta_median, free_names, fixed_params, datasets):
    """
    Calcula el desglose por dataset usando compute_objective_breakdown,
    es decir, la misma función interna que usa objective_function.
    """

    rows = []

    for data in datasets:
        print(f"\nEvaluando desglose dataset {data['id']:02d}: {data['name']}")

        breakdown = compute_objective_breakdown(
            theta=theta_median,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=data["x0"],
            t_rel=data["t_rel"],
            temp=data["temp"],
            Nadd=data["Nadd"],
            t_span=data["t_span"],
            sugars_profile=data["sugars_profile"],
            Et_final_exp=data["Et_final_exp"],
            penalty=PENALTY_COST,
        )

        sugar_error_mean = breakdown["sugar_error_mean"]
        ethanol_error = breakdown["ethanol_error"]
        objective_total = breakdown["objective_total"]
        Et_final_sim = breakdown["Et_final_sim"]

        Et_final_exp = float(data["Et_final_exp"])
        etoh_abs_error = abs(Et_final_sim - Et_final_exp)

        if abs(Et_final_exp) > 1e-8:
            etoh_rel_error = etoh_abs_error / abs(Et_final_exp)
        else:
            etoh_rel_error = np.nan

        sugar_residual_vector = breakdown["sugar_residual_vector"]

        if sugar_residual_vector is not None and np.all(np.isfinite(sugar_residual_vector)):
            sugar_rmse_norm = float(np.sqrt(np.mean(sugar_residual_vector ** 2)))
        else:
            sugar_rmse_norm = np.nan

        row = {
            "dataset_id": data["id"],
            "dataset_name": data["name"],
            "objective_total": objective_total,
            "sugar_error_mean": sugar_error_mean,
            "ethanol_error": ethanol_error,
            "sugar_rmse_norm": sugar_rmse_norm,
            "Et_final_exp": Et_final_exp,
            "Et_final_sim": Et_final_sim,
            "EtOH_abs_error": etoh_abs_error,
            "EtOH_rel_error": etoh_rel_error,
            "sugar_scale": breakdown.get("sugar_scale", np.nan),
            "ethanol_scale": breakdown.get("ethanol_scale", np.nan),
        }

        rows.append(row)

        print(f"  objective_total:   {objective_total:.8f}")
        print(f"  sugar_error_mean:  {sugar_error_mean:.8f}")
        print(f"  ethanol_error:     {ethanol_error:.8f}")
        print(f"  EtOH exp final:    {Et_final_exp:.4f}")
        print(f"  EtOH sim final:    {Et_final_sim:.4f}")
        print(f"  Error EtOH abs.:   {etoh_abs_error:.4f}")
        print(f"  Error EtOH rel.:   {100 * etoh_rel_error:.2f}%")

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("COSTO DE VALIDACIÓN USANDO PARÁMETROS MEDIANOS")
    print("Usando la función de costo original del código de optimización")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Construir datasets de validación
    # --------------------------------------------------------
    validation_datasets = build_validation_datasets()

    # --------------------------------------------------------
    # 2. Construir theta_median
    # --------------------------------------------------------
    free_names, theta_median = build_theta_median_from_samples(FREE_PARAM_SAMPLES)

    print("\nParámetros libres usados:")
    for name, value in zip(free_names, theta_median):
        print(f"  {name}: {value:.10f}")

    print("\nParámetros fijos usados:")
    for name, value in FIXED_PARAMS.items():
        print(f"  {name}: {value:.10f}")

    # --------------------------------------------------------
    # 3. Costo promedio del set de validación
    #    usando directamente objective_function_multi
    # --------------------------------------------------------
    validation_cost_mean = objective_function_multi(
        theta=theta_median,
        free_names=free_names,
        fixed_params=FIXED_PARAMS,
        datasets=validation_datasets,
        penalty=PENALTY_COST,
    )

    print("\n" + "=" * 80)
    print("RESULTADO GLOBAL")
    print("=" * 80)
    print(f"Costo promedio del set de validación:")
    print(f"  J_validación = {validation_cost_mean:.10f}")

    # --------------------------------------------------------
    # 4. Desglose por dataset usando compute_objective_breakdown
    # --------------------------------------------------------
    df_breakdown = compute_dataset_breakdowns(
        theta_median=theta_median,
        free_names=free_names,
        fixed_params=FIXED_PARAMS,
        datasets=validation_datasets,
    )

    print("\n" + "=" * 80)
    print("RESUMEN POR DATASET")
    print("=" * 80)

    print(
        df_breakdown[
            [
                "dataset_id",
                "dataset_name",
                "objective_total",
                "sugar_error_mean",
                "ethanol_error",
                "Et_final_exp",
                "Et_final_sim",
                "EtOH_abs_error",
                "EtOH_rel_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
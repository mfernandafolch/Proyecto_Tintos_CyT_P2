# -*- coding: utf-8 -*-
"""
Gráficos de incertidumbre para 4 datasets elegidos al azar
- Se eligen 4 de los 16 datasets antes de cargarlos
- Se simula con el promedio de parámetros
- Se construyen bandas con 100 simulaciones muestreadas
- Se grafica azúcares (G+F) y etanol en un mismo eje
- Figura final en formato 2x2
"""

import os
import sys
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation, simulate_system
from pymoo_opt import PARAM_ORDER


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

N_DATASETS_TO_PLOT = 4
N_MONTE_CARLO = 100
N_MONTE_CARLO_WORKERS = 4   # Si quieres serial, usa 1
SEED_DATASET_SELECTION = None

LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95


# ============================================================
# LISTA DE LOS 16 DATASETS 100.000 L
# ============================================================

DATASETS_INFO = [
    {
        "id": 1,
        "name": "Data CS 24 EL BOLDO estanque 30.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 EL BOLDO estanque 30.xlsx",
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
# PARÁMETROS ESTIMADOS Y FIJOS
# ============================================================

# FREE_PARAM_SAMPLES = {
#     "mu0": [
#         0.295411929, 0.297014527, 0.533773573, 0.471389584, 0.343196835,
#         0.380799596, 0.142087516, 0.465845354, 0.249348931, 0.470460407,
#         0.471740849, 0.324577319, 0.238574863, 0.204083214, 0.605232513,
#         0.15794396, 0.403636819, 0.19313678, 0.252934845, 0.426851603
#     ],
#     "betaG0": [
#         1.080712012, 2.512098003, 2.285402627, 2.449770162, 3.197019981,
#         2.320937909, 0.459905866, 2.295248634, 2.372804833, 3.150002935,
#         1.102587559, 2.319700056, 2.392300558, 2.407000078, 2.904414124,
#         2.355137149, 0.807629742, 2.238263563, 2.540497964, 2.360635614
#     ],
#     "betaF0": [
#         4.153414964, 2.779960659, 5.997210589, 4.988556796, 4.17234724,
#         1.822644048, 5.468148421, 4.96168605, 5.34130598, 6.672897945,
#         4.391512518, 3.369434653, 3.401605806, 5.239593338, 6.313410682,
#         5.10021909, 4.027143028, 1.466105855, 4.488571402, 5.414511459
#     ],
#     "Yxn": [
#         1.745039102, 1.71175839, 0.867881891, 1.038609207, 1.10832363,
#         2.529317183, 4.205276318, 1.101102385, 1.311327473, 0.691296406,
#         1.596560452, 1.692008454, 1.529744449, 1.137591756, 0.878490921,
#         1.118415254, 2.383885966, 3.038121996, 1.184006702, 0.919329572
#     ],
#     "Yxg": [
#         7.389187861, 9.085213555, 8.955822973, 4.577804506, 4.227848729,
#         7.898741687, 7.659911066, 4.873825349, 8.248374652, 3.70298829,
#         4.803379149, 3.973952461, 5.969584936, 5.287290884, 9.776594213,
#         8.286461038, 2.799187003, 5.109769674, 9.147167269, 5.706533844
#     ],
#     "Yeg": [
#         0.161936722, 0.38987209, 0.23209247, 0.261866101, 0.358200693,
#         0.386525565, 0.283368092, 0.281458215, 0.392785477, 0.285709958,
#         0.170812153, 0.354638236, 0.339182086, 0.241697779, 0.25216673,
#         0.25084903, 0.211713606, 0.43052186, 0.309339266, 0.186004467
#     ],
#     "Yef": [
#         0.724518619, 0.451747082, 0.718240177, 0.63704131, 0.486592373,
#         0.553385568, 0.511414531, 0.603363562, 0.414395511, 0.600884368,
#         0.834661984, 0.517544357, 0.499847367, 0.65252543, 0.753590037,
#         0.705318143, 0.566817513, 0.462779224, 0.537615999, 4.511779167
#     ],
# }

FREE_PARAM_SAMPLES = {
    "mu0": [
        0.410522357, 0.608108642, 0.396985809, 0.110031355, 0.374723123
    ],
    "betaG0": [
        2.900356602, 1.18332082, 2.719339668, 0.997286407, 1.639174628
    ],
    "betaF0": [
        4.836693709, 2.602747839, 5.970591671, 7.719059015, 1.698242259
    ],
    "Yxn": [
        1.006776652, 2.553943376, 0.854789274, 2.912857123, 3.00530952
    ],
    "Yxg": [
        4.835959892, 1.719813183, 4.809271529, 5.708429953, 9.999984926
    ],
    "Yeg": [
        0.254140038, 0.253441559, 0.233060327, 0.39927457, 0.390373131
    ],
    "Yef": [
        0.72206529, 0.753078041, 0.709896832, 0.463704181, 0.436033896
    ],
}

FIXED_PARAMS = {
    "Kn0": 0.009647,
    "Kg0": 8.551854,
    "Kf0": 7.165650,
    "Kig0": 44.150670,
    "Kie0": 42.528284,
    "Kd0": 0.0001,
    "Yxf": 1.642634,
}


# ============================================================
# UTILIDADES DE PARÁMETROS
# ============================================================

def compute_free_param_statistics(free_param_samples):
    free_param_mean = {}
    free_param_std = {}

    for name, values in free_param_samples.items():
        arr = np.asarray(values, dtype=float)

        if arr.ndim != 1:
            raise ValueError(f"Los valores de '{name}' deben ser una lista 1D.")
        if len(arr) < 2:
            raise ValueError(
                f"'{name}' debe tener al menos 2 valores para calcular desviación estándar."
            )

        free_param_mean[name] = float(np.mean(arr))
        free_param_std[name] = float(np.std(arr, ddof=1))

    return free_param_mean, free_param_std


FREE_PARAM_MEAN, FREE_PARAM_STD = compute_free_param_statistics(FREE_PARAM_SAMPLES)


def build_mean_param_dict():
    params = FIXED_PARAMS.copy()
    params.update(FREE_PARAM_MEAN)
    return params


def sample_positive_normal(mean_value, std_value, max_tries=1000):
    if std_value <= 0:
        return float(mean_value)

    for _ in range(max_tries):
        value = np.random.normal(loc=mean_value, scale=std_value)
        if value > 0:
            return float(value)

    return float(max(mean_value, 1e-8))


def sample_free_params():
    sampled = {}
    for name in FREE_PARAM_MEAN:
        sampled[name] = sample_positive_normal(
            FREE_PARAM_MEAN[name],
            FREE_PARAM_STD[name]
        )
    return sampled


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def choose_random_datasets(datasets_info, n_to_choose, seed=None):
    rng = random.Random(seed) if seed is not None else random
    selected = rng.sample(datasets_info, n_to_choose)
    return sorted(selected, key=lambda x: x["id"])


def build_dataset(item):
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


def build_param_vector(param_dict):
    return np.array([param_dict[name] for name in PARAM_ORDER], dtype=float)


def simulate_dataset(dataset, params_dict):
    params_vector = build_param_vector(params_dict)

    sol = simulate_system(
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector
    )

    y = sol.y.T
    sugars = np.asarray(y[:, 2] + y[:, 3], dtype=float)
    ethanol = np.asarray(y[:, 4], dtype=float)

    return {
        "time": np.asarray(sol.t, dtype=float),
        "sugars": sugars,
        "ethanol": ethanol,
    }


# ============================================================
# MONTE CARLO
# ============================================================

def run_single_monte_carlo_iteration(dataset):
    sampled_params = FIXED_PARAMS.copy()
    sampled_params.update(sample_free_params())

    try:
        sim = simulate_dataset(dataset, sampled_params)
        return sim["sugars"], sim["ethanol"]
    except Exception:
        return None


def run_uncertainty_simulations(dataset, n_mc, n_workers=None):
    mean_params = build_mean_param_dict()
    base_sim = simulate_dataset(dataset, mean_params)

    sugar_runs = []
    ethanol_runs = []

    if n_workers == 1:
        for _ in range(n_mc):
            mc_result = run_single_monte_carlo_iteration(dataset)
            if mc_result is None:
                continue
            sugars, ethanol = mc_result
            sugar_runs.append(sugars)
            ethanol_runs.append(ethanol)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunksize = max(1, n_mc // 10)
            for mc_result in executor.map(
                run_single_monte_carlo_iteration,
                repeat(dataset, n_mc),
                chunksize=chunksize,
            ):
                if mc_result is None:
                    continue
                sugars, ethanol = mc_result
                sugar_runs.append(sugars)
                ethanol_runs.append(ethanol)

    if len(sugar_runs) == 0 or len(ethanol_runs) == 0:
        raise RuntimeError(
            f"No se pudieron generar simulaciones válidas para el dataset {dataset['name']}"
        )

    sugar_runs = np.asarray(sugar_runs, dtype=float)
    ethanol_runs = np.asarray(ethanol_runs, dtype=float)

    sugar_low = np.percentile(sugar_runs, LOW_PERCENTILE, axis=0)
    sugar_high = np.percentile(sugar_runs, HIGH_PERCENTILE, axis=0)

    ethanol_low = np.percentile(ethanol_runs, LOW_PERCENTILE, axis=0)
    ethanol_high = np.percentile(ethanol_runs, HIGH_PERCENTILE, axis=0)

    return {
        "time": base_sim["time"],
        "sugars_mean_curve": base_sim["sugars"],
        "ethanol_mean_curve": base_sim["ethanol"],
        "sugar_low": sugar_low,
        "sugar_high": sugar_high,
        "ethanol_low": ethanol_low,
        "ethanol_high": ethanol_high,
        "n_valid_runs": len(sugar_runs),
    }


# ============================================================
# PLOT
# ============================================================

def plot_results(datasets, results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, dataset, res in zip(axes, datasets, results):
        t_sim_days = res["time"] / 24.0
        t_exp_days = np.asarray(dataset["t_rel"], dtype=float) / 24.0

        ax.fill_between(
            t_sim_days,
            res["sugar_low"],
            res["sugar_high"],
            alpha=0.20,
            label="Banda azúcares"
        )
        ax.plot(
            t_sim_days,
            res["sugars_mean_curve"],
            linewidth=2.0,
            label="Azúcares simulados"
        )
        ax.scatter(
            t_exp_days,
            dataset["sugars_profile"],
            s=22,
            label="Azúcares experimentales"
        )

        ax.fill_between(
            t_sim_days,
            res["ethanol_low"],
            res["ethanol_high"],
            alpha=0.20,
            label="Banda etanol"
        )
        ax.plot(
            t_sim_days,
            res["ethanol_mean_curve"],
            linewidth=2.0,
            linestyle="--",
            label="Etanol simulado"
        )
        ax.scatter(
            t_exp_days[-1],
            dataset["Et_final_exp"],
            s=45,
            marker="o",
            label="Etanol final experimental"
        )

        ax.set_title(
            f"Set {dataset['id']}: {dataset['name']}\n"
            f"Simulaciones válidas: {res['n_valid_runs']}/{N_MONTE_CARLO}",
            fontsize=10
        )
        ax.set_xlabel("Tiempo (días)")
        ax.set_ylabel("Concentración")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98)
    )

    fig.suptitle(
        "Incertidumbre de simulación para azúcares y etanol\n"
        "Curva central con parámetros promedio + banda percentil 5-95",
        fontsize=14,
        y=1.03
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("GRÁFICOS DE INCERTIDUMBRE - 4 DATASETS ALEATORIOS")
    print("=" * 80)

    print("\nPromedios calculados:")
    for k, v in FREE_PARAM_MEAN.items():
        print(f"{k}: {v:.8f}")

    print("\nDesviaciones estándar calculadas:")
    for k, v in FREE_PARAM_STD.items():
        print(f"{k}: {v:.8f}")

    selected_info = choose_random_datasets(
        DATASETS_INFO,
        n_to_choose=N_DATASETS_TO_PLOT,
        seed=SEED_DATASET_SELECTION
    )

    print("\nDatasets elegidos:")
    for item in selected_info:
        print(f"  {item['id']:02d} - {item['name']}")

    datasets = []
    for item in selected_info:
        print(f"\nCargando dataset {item['id']:02d}...")
        datasets.append(build_dataset(item))

    results = []
    for dataset in datasets:
        print(f"\nCorriendo simulaciones para set {dataset['id']:02d} - {dataset['name']}")
        res = run_uncertainty_simulations(
            dataset,
            N_MONTE_CARLO,
            n_workers=N_MONTE_CARLO_WORKERS
        )
        results.append(res)

    plot_results(datasets, results)


if __name__ == "__main__":
    main()
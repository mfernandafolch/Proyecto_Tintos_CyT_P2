# -*- coding: utf-8 -*-

import os
import sys
import textwrap
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation, simulate_system
from pymoo_opt import PARAM_ORDER


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

DATASET_IDS_TO_PLOT = [3, 4, 11, 14]

TITLE_WRAP_WIDTH = 42


# ============================================================
# 2. PARÁMETROS LIBRES Y FIJOS
# ============================================================

FREE_PARAMS = {
    "mu0": 0.257,
    "betaG0": 2.037,
    "betaF0": 3.042,
    "Yxn": 3.639,
    "Yxg": 7.096,
    "Yeg": 0.432,
    "Yef": 0.444,
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
# 3. DATASETS DISPONIBLES
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
# 4. FUNCIONES AUXILIARES
# ============================================================

def choose_datasets_by_ids(datasets_info, dataset_ids):
    dataset_map = {item["id"]: item for item in datasets_info}

    missing = [i for i in dataset_ids if i not in dataset_map]
    if missing:
        raise ValueError(f"Estos IDs no existen en DATASETS_INFO: {missing}")

    return [dataset_map[i] for i in dataset_ids]


def build_param_vector(free_params, fixed_params):
    all_params = {}
    all_params.update(fixed_params)
    all_params.update(free_params)

    missing = [p for p in PARAM_ORDER if p not in all_params]
    extra = [p for p in all_params if p not in PARAM_ORDER]

    if missing:
        raise ValueError(f"Faltan parámetros: {missing}")

    if extra:
        raise ValueError(f"Hay parámetros no reconocidos: {extra}")

    return np.array([all_params[p] for p in PARAM_ORDER], dtype=float)


def build_dataset(item):
    data_excel = data_for_simulation(item["path"])

    return {
        "id": item["id"],
        "name": item["name"],
        "path": item["path"],
        "x0": data_excel[0],
        "t_rel": np.asarray(data_excel[1], dtype=float),
        "sugars_profile": np.asarray(data_excel[2], dtype=float),
        "temp": data_excel[3],
        "Nadd": data_excel[4],
        "t_span": data_excel[5],
        "Et_final_exp": float(data_excel[6]),
    }


def simulate_dataset(dataset):
    params_vector = build_param_vector(FREE_PARAMS, FIXED_PARAMS)

    sol = simulate_system(
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector,
    )

    y = sol.y.T

    return {
        "time": np.asarray(sol.t, dtype=float),
        "X": np.asarray(y[:, 0], dtype=float),
        "N": np.asarray(y[:, 1], dtype=float),
        "G": np.asarray(y[:, 2], dtype=float),
        "F": np.asarray(y[:, 3], dtype=float),
        "E": np.asarray(y[:, 4], dtype=float),
        "sugars": np.asarray(y[:, 2] + y[:, 3], dtype=float),
    }


def compute_plot_cost(dataset, sim):
    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    sugars_exp = np.asarray(dataset["sugars_profile"], dtype=float)

    sugars_sim_interp = np.interp(t_exp, sim["time"], sim["sugars"])

    valid = np.isfinite(sugars_exp) & np.isfinite(sugars_sim_interp)

    if np.any(valid):
        scale_sugar = max(np.nanmax(np.abs(sugars_exp[valid])), 1e-8)
        sugar_cost = np.mean(
            ((sugars_sim_interp[valid] - sugars_exp[valid]) / scale_sugar) ** 2
        )
    else:
        sugar_cost = 0.0

    ethanol_exp = float(dataset["Et_final_exp"])
    ethanol_sim_final = float(sim["E"][-1])

    scale_ethanol = max(abs(ethanol_exp), 1e-8)
    ethanol_cost = ((ethanol_sim_final - ethanol_exp) / scale_ethanol) ** 2

    return float(sugar_cost + ethanol_cost)


# ============================================================
# 5. FUNCIÓN DE GRÁFICO
# ============================================================

def plot_single_dataset(dataset, sim, cost=None):
    t_sim_days = sim["time"] / 24.0
    t_exp_days = dataset["t_rel"] / 24.0

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    ax.plot(
        t_sim_days,
        sim["sugars"],
        linewidth=2.0,
        label="Azúcares simulados",
    )

    ax.scatter(
        t_exp_days,
        dataset["sugars_profile"],
        s=32,
        label="Azúcares experimentales",
        zorder=3,
    )

    ax.plot(
        t_sim_days,
        sim["E"],
        linewidth=2.0,
        linestyle="--",
        label="Etanol simulado",
    )

    ax.scatter(
        t_exp_days[-1],
        dataset["Et_final_exp"],
        s=60,
        marker="o",
        label="Etanol final experimental",
        zorder=4,
    )

    if cost is not None:
        ax.text(
            0.60,
            0.94,
            f"Costo: {cost:.6f}",
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    title = f"Set {dataset['id']}: {dataset['name']}"
    ax.set_title(
        textwrap.fill(title, width=TITLE_WRAP_WIDTH),
        fontsize=11,
        pad=14,
    )

    ax.set_xlabel("Tiempo (días)", labelpad=8)
    ax.set_ylabel("Concentración (g/L)", labelpad=8)

    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=9,
        frameon=True,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.25,
        top=0.84,
    )

    plt.show()


# ============================================================
# 6. MAIN
# ============================================================

def main():
    selected_info = choose_datasets_by_ids(
        DATASETS_INFO,
        DATASET_IDS_TO_PLOT,
    )

    print("=" * 80)
    print("GRÁFICOS DE SIMULACIÓN CON SET PUNTUAL DE PARÁMETROS")
    print("=" * 80)

    print("\nDatasets seleccionados:")
    for item in selected_info:
        print(f"  Set {item['id']:02d}: {item['name']}")

    print("\nParámetros libres:")
    for key, value in FREE_PARAMS.items():
        print(f"  {key}: {value}")

    print("\nParámetros fijos:")
    for key, value in FIXED_PARAMS.items():
        print(f"  {key}: {value}")

    for item in selected_info:
        print("\n" + "-" * 80)
        print(f"Cargando set {item['id']:02d} - {item['name']}")

        dataset = build_dataset(item)

        print("Simulando...")
        sim = simulate_dataset(dataset)

        cost = compute_plot_cost(dataset, sim)

        print(f"Costo gráfico: {cost:.6f}")
        print(f"Etanol final experimental: {dataset['Et_final_exp']:.4f}")
        print(f"Etanol final simulado:     {sim['E'][-1]:.4f}")

        plot_single_dataset(dataset, sim, cost=cost)


if __name__ == "__main__":
    main()
"""
main_plot_uncertainty.py

Validación con incertidumbre Monte Carlo para el modelo original de Coleman:
1) Azúcares totales S
2) Etanol E

Cada figura tiene 4 subplots, uno por cada dataset de validación.

- Curva negra: simulación con mediana de los parámetros.
- Todas las simulaciones Monte Carlo: líneas tenues superpuestas.
- Banda roja: envolvente entre la simulación mínima y la máxima.
- Puntos azules: datos experimentales.

El muestreo de parámetros libres se puede configurar para usar una ventana de
±1σ o ±2σ alrededor de la mediana.
"""

import os
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion_coleman import data_for_simulation, simulate_system
from pymoo_opt_coleman import PARAM_ORDER, compute_objective_breakdown


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VALIDATION_DATASET_IDS = [3, 4, 11, 14]

DATASET_MARKERS = ["o", "s", "^", "*"]

N_MONTE_CARLO = 100
N_MONTE_CARLO_WORKERS = 4

# Ventana de muestreo Monte Carlo alrededor de la mediana: 1 o 2 desviaciones estándar.
MC_STD_WINDOW = 2

RANDOM_SEED = 123
TITLE_WRAP_WIDTH = 42
PENALTY_COST = 1e12


# ============================================================
# BOUNDS DE PARÁMETROS
# ============================================================

BOUNDS_DICT = {
    "mu0": (1e-3, 100.0),
    "kd0": (1e-3, 100.0),
    "betaS0": (1e-3, 100.0),
    "Kn": (1e-5, 100.0),
    "Yxn": (1e-2, 100.0),
    "Yes": (1e-2, 10.0),
    "Ks": (1e-2, 100.0),
}


# ============================================================
# LISTA DE DATASETS
# ============================================================

DATASETS_INFO = [
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
        "id": 11,
        "name": "Data ME 25 AURORA + STA MARTA estanque 57.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 AURORA + STA MARTA estanque 57.xlsx",
    },
    {
        "id": 14,
        "name": "Data CA 24 VAL estanque 59.xlsx",
        "path": r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 59.xlsx",
    },
]


# ============================================================
# PARÁMETROS ESTIMADOS Y FIJOS
# ============================================================

FREE_PARAM_SAMPLES = {
    "Kn": [0.000335154,0.002412383, 3.26013E-05, 0.001000834, 1.89649E-05, 
           1E-05, 1.21529E-05, 0.000284886, 0.001000651, 0.001033548, 
           0.001480081, 2.41554E-05, 0.000469009, 2.24236E-05, 0.001005548,
           0.00100316, 0.001142938, 0.001030992, 0.001025347,
           0.00028489
           ],

    "Yxn": [27.52949875, 19.25269315, 28.10216922, 19.60166733, 34.88636949,
            21.01922919, 18.03651243, 26.20377448, 17.68210745, 18.68294675,
            23.4141672, 25.50523211, 38.68924627, 31.02966681, 9.999783247, 
            9.999867119, 9.99999984, 9.996601467, 9.999981646,
            23.41416720
            ],

    "Yes": [0.364039663, 0.400808166, 0.414406585, 0.361245948, 0.373919816, 
            0.398394725, 0.367509843, 0.394678147, 0.388410172, 0.373000203, 
            0.380704243,0.375858325, 0.363191521, 0.370174245, 0.299081104, 
            0.295197201, 0.295341656, 0.314331362, 0.282601987,
            0.373919816
            ],

    "Ks": [65.43314466, 10.66808801, 99.99844878, 19.60016407, 49.35264987, 
           39.80042928, 0.063086977, 51.67054352, 14.93618506, 13.75091923, 
           12.37472628, 47.22759253, 81.81196442, 49.55543244, 0.026957544, 
           0.020414208, 0.010525965, 0.010108712, 0.010658929,
           39.80042928
           ],}


FIXED_PARAMS = {
    "mu0": 1.0,
    "kd0": 1.0,
    "betaS0": 1.0,
}

# ============================================================
# UTILIDADES DE PARÁMETROS
# ============================================================

def compute_free_param_statistics(free_param_samples):
    free_param_median = {}
    free_param_std = {}

    for name, values in free_param_samples.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.ndim != 1:
            raise ValueError(f"Los valores de '{name}' deben ser una lista 1D.")

        if len(arr) < 2:
            raise ValueError(
                f"'{name}' debe tener al menos 2 valores válidos para calcular desviación estándar."
            )

        if name not in BOUNDS_DICT:
            raise ValueError(
                f"El parámetro libre '{name}' no está en BOUNDS_DICT. "
                "Agrega sus bounds antes de muestrearlo."
            )

        free_param_median[name] = float(np.nanmedian(arr))
        free_param_std[name] = float(np.nanstd(arr, ddof=1))

    return free_param_median, free_param_std


def build_free_param_matrix(free_param_samples):
    """
    Construye una matriz con los parámetros libres.

    A diferencia de la versión anterior, ahora permite que cada parámetro tenga
    una cantidad distinta de muestras. Para poder armar una matriz rectangular,
    las columnas más cortas se rellenan con NaN. Esta matriz queda disponible
    por si quieres inspeccionarla, pero las estadísticas se calculan parámetro
    a parámetro, ignorando NaN.
    """

    free_names = list(free_param_samples.keys())
    lengths = {name: len(free_param_samples[name]) for name in free_names}
    max_len = max(lengths.values())

    matrix = np.full((max_len, len(free_names)), np.nan, dtype=float)

    for j, name in enumerate(free_names):
        arr = np.asarray(free_param_samples[name], dtype=float)
        matrix[:len(arr), j] = arr

    return free_names, matrix


def get_param_sample_counts(free_param_samples):
    """Retorna un diccionario con la cantidad de muestras disponibles por parámetro."""
    return {name: len(values) for name, values in free_param_samples.items()}


def format_param_sample_counts(sample_counts):
    """Texto para títulos: Kn n=13, Yxn n=10, ..."""
    return ", ".join([f"{name} n={n}" for name, n in sample_counts.items()])


FREE_PARAM_MEDIAN, FREE_PARAM_STD = compute_free_param_statistics(FREE_PARAM_SAMPLES)
FREE_PARAM_NAMES, FREE_PARAM_MATRIX = build_free_param_matrix(FREE_PARAM_SAMPLES)
FREE_PARAM_SAMPLE_COUNTS = get_param_sample_counts(FREE_PARAM_SAMPLES)
FREE_PARAM_SAMPLE_COUNTS_TEXT = format_param_sample_counts(FREE_PARAM_SAMPLE_COUNTS)


def build_median_param_dict():
    params = FIXED_PARAMS.copy()
    params.update(FREE_PARAM_MEDIAN)

    missing_params = [name for name in PARAM_ORDER if name not in params]
    if missing_params:
        raise ValueError(
            "Faltan parámetros para construir el vector completo según PARAM_ORDER: "
            f"{missing_params}"
        )

    return params


def sample_truncated_normal_parameter(name, rng):
    return sample_truncated_normal_parameter_with_window(name, rng, MC_STD_WINDOW)


def sample_truncated_normal_parameter_with_window(name, rng, sigma_window):
    median_value = FREE_PARAM_MEDIAN[name]
    std_value = FREE_PARAM_STD[name]
    lb, ub = BOUNDS_DICT[name]

    median_value = float(np.clip(median_value, lb, ub))

    if sigma_window not in (1, 2):
        raise ValueError("MC_STD_WINDOW debe ser 1 o 2.")

    local_lb = max(lb, median_value - sigma_window * std_value)
    local_ub = min(ub, median_value + sigma_window * std_value)

    if local_ub <= local_lb:
        return median_value

    if not np.isfinite(std_value) or std_value <= 0:
        return median_value

    a = (local_lb - median_value) / std_value
    b = (local_ub - median_value) / std_value

    sampled_value = truncnorm.rvs(
        a=a,
        b=b,
        loc=median_value,
        scale=std_value,
        random_state=rng
    )

    return float(sampled_value)


def sample_free_params_truncnorm(seed=None, sigma_window=MC_STD_WINDOW):
    rng = np.random.default_rng(seed)

    sampled = {}
    for name in FREE_PARAM_NAMES:
        sampled[name] = sample_truncated_normal_parameter_with_window(
            name,
            rng,
            sigma_window,
        )

    return sampled


def build_sampled_param_dict(seed=None, sigma_window=MC_STD_WINDOW):
    params = FIXED_PARAMS.copy()
    params.update(sample_free_params_truncnorm(seed=seed, sigma_window=sigma_window))

    missing_params = [name for name in PARAM_ORDER if name not in params]
    if missing_params:
        raise ValueError(
            "Faltan parámetros para construir el vector completo según PARAM_ORDER: "
            f"{missing_params}"
        )

    return params


def build_param_vector(param_dict):
    return np.array([param_dict[name] for name in PARAM_ORDER], dtype=float)


def build_median_theta_vector():
    """Vector de parámetros libres en el mismo orden que FREE_PARAM_NAMES."""
    return np.array([FREE_PARAM_MEDIAN[name] for name in FREE_PARAM_NAMES], dtype=float)


def compute_validation_costs(dataset):
    """
    Calcula los costos con la misma función usada en la optimización.

    Retorna:
    - objective_total: costo total azúcar + etanol
    - sugar_error_mean: término de costo de azúcares
    - ethanol_error: término de costo de etanol
    """

    breakdown = compute_objective_breakdown(
        theta=build_median_theta_vector(),
        free_names=FREE_PARAM_NAMES,
        fixed_params=FIXED_PARAMS,
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        t_span=dataset["t_span"],
        sugars_profile=dataset["sugars_profile"],
        Et_final_exp=dataset["Et_final_exp"],
        penalty=PENALTY_COST,
    )

    return {
        "validation_cost_total": float(breakdown["objective_total"]),
        "validation_cost_sugar": float(breakdown["sugar_error_mean"]),
        "validation_cost_ethanol": float(breakdown["ethanol_error"]),
    }


# ============================================================
# UTILIDADES DE DATASETS
# ============================================================

def choose_datasets_by_ids(datasets_info, dataset_ids):
    if len(dataset_ids) == 0:
        raise ValueError("Debes entregar al menos un ID de dataset.")

    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(f"Hay IDs repetidos en VALIDATION_DATASET_IDS: {dataset_ids}")

    dataset_map = {item["id"]: item for item in datasets_info}

    missing_ids = [
        dataset_id for dataset_id in dataset_ids
        if dataset_id not in dataset_map
    ]

    if missing_ids:
        raise ValueError(
            f"Los siguientes IDs no existen en DATASETS_INFO: {missing_ids}"
        )

    return [dataset_map[dataset_id] for dataset_id in dataset_ids]


def build_dataset(item):
    data_excel = data_for_simulation(item["path"])
    sugar_initial = data_excel[2][0] if len(data_excel) > 2 and len(data_excel[2]) > 0 else None

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
        "sugar_initial": float(sugar_initial) if sugar_initial is not None else None,
    }


# ============================================================
# SIMULACIÓN
# ============================================================

def simulate_dataset(dataset, params_dict):
    """
    Simula un dataset y retorna:
    - tiempo
    - azúcares totales S
    - etanol E
    """

    params_vector = build_param_vector(params_dict)
    x0_og = np.asarray(dataset["x0"], dtype=float)

    # El modelo Coleman aquí trabaja con 4 estados: [X, N, S, E]
    X0 = x0_og[0]
    N0 = x0_og[1]
    E0 = x0_og[3]

    # Cambiar azúcares
    if dataset["sugar_initial"] is not None:
        S0 = float(dataset["sugar_initial"])
    else:
        S0 = 0.0

    x0 = np.array([X0, N0, S0, E0], dtype=float)

    sol = simulate_system(
        x0=x0,
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector
    )

    y = sol.y.T

    sugars = np.asarray(y[:, 2], dtype=float)
    ethanol = np.asarray(y[:, 3], dtype=float)

    if not np.all(np.isfinite(sugars)):
        raise RuntimeError("La simulación produjo valores no finitos en azúcares.")

    if not np.all(np.isfinite(ethanol)):
        raise RuntimeError("La simulación produjo valores no finitos en etanol.")

    return {
        "time": np.asarray(sol.t, dtype=float),
        "sugars": sugars,
        "ethanol": ethanol,
    }


# ============================================================
# MÉTRICAS
# ============================================================

def compute_sugar_validation_metrics(dataset, result):
    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)

    t_sim = np.asarray(result["time"], dtype=float)
    sugar_central = np.asarray(result["sugars_central"], dtype=float)

    sugar_interp = np.interp(t_exp, t_sim, sugar_central)

    valid = np.isfinite(t_exp) & np.isfinite(sugar_exp) & np.isfinite(sugar_interp)

    if not np.any(valid):
        return {
            "rmse": np.nan,
            "nrmse": np.nan,
            "coverage": np.nan,
            "n_exp_valid": 0,
        }

    y_exp = sugar_exp[valid]
    y_sim = sugar_interp[valid]

    # RMSE = sqrt(1/n * sum((y_i - yhat_i)^2))
    errors = y_exp - y_sim
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # NRMSE = RMSE / (y_max - y_min)
    y_range = float(np.nanmax(y_exp) - np.nanmin(y_exp))
    nrmse = float(rmse / y_range) if y_range > 1e-8 else np.nan

    low_interp = np.interp(t_exp[valid], t_sim, result["sugar_min_max_bands"]["min"])
    high_interp = np.interp(t_exp[valid], t_sim, result["sugar_min_max_bands"]["max"])

    inside = (y_exp >= low_interp) & (y_exp <= high_interp)
    coverage = float(100.0 * np.mean(inside))

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "coverage": coverage,
        "n_exp_valid": int(np.sum(valid)),
    }

def compute_ethanol_validation_metrics(dataset, result):
    et_exp = float(dataset["Et_final_exp"])
    et_central_final = float(result["ethanol_central"][-1])

    error = et_central_final - et_exp
    abs_error = abs(error)

    scale = max(abs(et_exp), 1e-8)
    relative_error = abs_error / scale

    return {
        "error": error,
        "abs_error": abs_error,
        "relative_error": relative_error,
    }


# ============================================================
# MONTE CARLO
# ============================================================

def run_single_monte_carlo_iteration(dataset, seed):
    """
    Ejecuta una simulación Monte Carlo.
    Retorna azúcares y etanol.
    """

    try:
        sampled_params = build_sampled_param_dict(seed=seed)
        sim = simulate_dataset(dataset, sampled_params)

        return {
            "sugars": sim["sugars"],
            "ethanol": sim["ethanol"],
        }

    except Exception:
        return None


def compute_min_max_envelope(runs):
    """Calcula la envolvente completa entre el mínimo y el máximo de las corridas."""

    runs = np.asarray(runs, dtype=float)

    return {
        "min": np.min(runs, axis=0),
        "max": np.max(runs, axis=0),
    }


def run_uncertainty_simulations(dataset, n_mc, n_workers=1):
    """
    Para un dataset:
    - simula curva central con mediana de los parámetros
    - genera bandas Monte Carlo para azúcares y etanol
    """

    median_params = build_median_param_dict()
    central_sim = simulate_dataset(dataset, median_params)

    rng = np.random.default_rng(RANDOM_SEED + int(dataset["id"]) * 1000)
    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=n_mc,
        dtype=np.uint32
    )

    sugar_runs = []
    ethanol_runs = []

    if n_workers == 1:
        for seed in seeds:
            mc_result = run_single_monte_carlo_iteration(dataset, int(seed))

            if mc_result is None:
                continue

            sugar_runs.append(mc_result["sugars"])
            ethanol_runs.append(mc_result["ethanol"])

    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunksize = max(1, n_mc // 10)

            for mc_result in executor.map(
                run_single_monte_carlo_iteration,
                repeat(dataset, n_mc),
                [int(seed) for seed in seeds],
                chunksize=chunksize,
            ):
                if mc_result is None:
                    continue

                sugar_runs.append(mc_result["sugars"])
                ethanol_runs.append(mc_result["ethanol"])

    if len(sugar_runs) == 0:
        raise RuntimeError(
            f"No se pudieron generar simulaciones válidas para el dataset {dataset['name']}."
        )

    sugar_runs = np.asarray(sugar_runs, dtype=float)
    ethanol_runs = np.asarray(ethanol_runs, dtype=float)

    sugar_bands = compute_min_max_envelope(sugar_runs)
    ethanol_bands = compute_min_max_envelope(ethanol_runs)

    result = {
        "time": central_sim["time"],

        "sugars_central": central_sim["sugars"],
        "sugar_runs": sugar_runs,
        "sugar_min_max_bands": sugar_bands,

        "ethanol_central": central_sim["ethanol"],
        "ethanol_runs": ethanol_runs,
        "ethanol_min_max_bands": ethanol_bands,

        "n_valid_runs": len(sugar_runs),
    }

    result["validation_costs"] = compute_validation_costs(dataset)
    result["sugar_metrics"] = compute_sugar_validation_metrics(dataset, result)
    result["ethanol_metrics"] = compute_ethanol_validation_metrics(dataset, result)

    return result


# ============================================================
# PLOT AUXILIAR
# ============================================================

def plot_all_mc_runs(ax, t_days, runs, color="#c04a4a"):
    """Dibuja todas las simulaciones con transparencia baja."""

    for run in runs:
        ax.plot(t_days, run, color=color, alpha=0.10, linewidth=0.9, zorder=1)


def add_min_max_envelope(ax, t_days, bands, label_first=True):
    """Dibuja la envolvente completa entre el mínimo y el máximo."""

    ax.fill_between(
        t_days,
        bands["min"],
        bands["max"],
        color="#d45d5d",
        alpha=0.24,
        linewidth=0,
        label="Envolvente min-max" if label_first else None,
        zorder=0,
    )


def create_2x2_axes(figsize=(16, 10.8)):
    fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        sharex=False,
        sharey=False
    )
    return fig, axes.flatten()


def clean_dataset_name(name):
    """Quita la extensión .xlsx del nombre mostrado en los gráficos."""
    return os.path.splitext(name)[0]


# ============================================================
# FIGURA 1: AZÚCARES
# ============================================================

def plot_sugar_results(datasets, results):
    fig, axes = create_2x2_axes()

    for idx, (ax, dataset, res) in enumerate(zip(axes, datasets, results)):
        t_sim_days = np.asarray(res["time"], dtype=float) / 24.0
        t_exp_days = np.asarray(dataset["t_rel"], dtype=float) / 24.0
        sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)
        valid_exp = np.isfinite(t_exp_days) & np.isfinite(sugar_exp)

        plot_all_mc_runs(ax, t_sim_days, res["sugar_runs"])
        add_min_max_envelope(ax, t_sim_days, res["sugar_min_max_bands"], label_first=(idx == 0))

        ax.plot(
            t_sim_days,
            res["sugars_central"],
            color="black",
            linewidth=2.2,
            marker=DATASET_MARKERS[idx],
            markersize=6.5,
            markerfacecolor="black",
            markeredgecolor="none",
            markeredgewidth=0,
            label="Simulación con la mediana de los parámetros",
            zorder=4,
        )

        ax.scatter(
            t_exp_days[valid_exp],
            sugar_exp[valid_exp],
            s=50,
            color="tab:blue",
            marker=DATASET_MARKERS[idx],
            zorder=3,
            label="Azúcares experimentales",
        )

        metrics = res["sugar_metrics"]
        costs = res["validation_costs"]
        text_box = (
            f"RMSE: {metrics['rmse']:.2f} g/L\n"
            f"NRMSE: {100 * metrics['nrmse']:.2f}%\n"
            f"Cobertura: {metrics['coverage']:.1f}%\n"
            f"Costo azúcar: {costs['validation_cost_sugar']:.4f}\n"
            f"Muestreo: ±{MC_STD_WINDOW}σ"
        )

        ax.text(0.04, 0.55, text_box, transform=ax.transAxes, fontsize=8.0,
                va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))

        ax.set_title(
            f"Set {dataset['id']:02d} - {textwrap.fill(clean_dataset_name(dataset['name']), width=TITLE_WRAP_WIDTH)}",
            # f"Costo validación total: {costs['validation_cost_total']:.6f}",
            fontsize=10,
            pad=12,
        )
        ax.set_xlabel("Tiempo (días)", labelpad=8)
        ax.set_ylabel("Azúcares, S (g/L)", labelpad=8)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(
        "Validación predictiva del consumo de azúcares - modelo Coleman\n"
        "Curva central con mediana de parámetros; 20 datos por parámetro\n"
        f"Bandas con {N_MONTE_CARLO} muestras Monte Carlo "
        f"(muestreo truncado en mediana ±{MC_STD_WINDOW}σ)",
        fontsize=12,
        y=0.985,
    )
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.905), fontsize=9.5, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.80, hspace=0.4, wspace=0.194)
    plt.show()


# ============================================================
# FIGURA 2: ETANOL
# ============================================================

def plot_ethanol_results(datasets, results):
    fig, axes = create_2x2_axes()

    for idx, (ax, dataset, res) in enumerate(zip(axes, datasets, results)):
        t_sim_days = np.asarray(res["time"], dtype=float) / 24.0
        et_exp = float(dataset["Et_final_exp"])
        t_final_exp_days = float(np.nanmax(np.asarray(dataset["t_rel"], dtype=float))) / 24.0

        plot_all_mc_runs(ax, t_sim_days, res["ethanol_runs"])
        add_min_max_envelope(ax, t_sim_days, res["ethanol_min_max_bands"], label_first=(idx == 0))

        ax.plot(
            t_sim_days,
            res["ethanol_central"],
            color="black",
            linewidth=2.2,
            marker=DATASET_MARKERS[idx],
            markersize=6.5,
            markerfacecolor="black",
            markeredgecolor="none",
            markeredgewidth=0,
            label="Simulación con la mediana de los parámetros",
            zorder=4,
        )

        ax.scatter([t_final_exp_days], [et_exp], s=50, color="tab:blue",
                   marker=DATASET_MARKERS[idx], zorder=4, label="Etanol experimental final")

        metrics = res["ethanol_metrics"]
        costs = res["validation_costs"]
        text_box = (
            f"Error abs.: {metrics['abs_error']:.2f} g/L\n"
            f"Error rel.: {100 * metrics['relative_error']:.2f}%\n"
            f"Costo etanol: {costs['validation_cost_ethanol']:.4f}\n"
            f"Muestreo: ±{MC_STD_WINDOW}σ"
        )

        ax.text(0.04, 0.94, text_box, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))

        ax.set_title(
            f"Set {dataset['id']:02d} - {textwrap.fill(clean_dataset_name(dataset['name']), width=TITLE_WRAP_WIDTH)}",
            # f"Costo validación total: {costs['validation_cost_total']:.6f}",
            fontsize=10,
            pad=12,
        )
        ax.set_xlabel("Tiempo (días)", labelpad=8)
        ax.set_ylabel("Etanol, E (g/L)", labelpad=8)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(
        "Validación predictiva de etanol final - modelo Coleman\n"
        "Curva central con mediana de parámetros; 20 datos por parámetro\n"
        f"Bandas con {N_MONTE_CARLO} muestras Monte Carlo "
        f"(muestreo truncado en mediana ±{MC_STD_WINDOW}σ)",
        fontsize=12,
        y=0.985,
    )
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.905), fontsize=9.5, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.80, hspace=0.4, wspace=0.194)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("VALIDACIÓN CON BANDAS MONTE CARLO - AZÚCARES Y ETANOL")
    print("=" * 80)

    print("\nConfiguración básica:")
    print(f"  Datasets validación: {VALIDATION_DATASET_IDS}")
    print(f"  Muestras Monte Carlo por dataset: {N_MONTE_CARLO}")
    print(f"  Workers: {N_MONTE_CARLO_WORKERS}")
    print(f"  Ventana de muestreo: ±{MC_STD_WINDOW} desviaciones estándar")

    print("\nMuestras disponibles por parámetro libre:")
    for name, n in FREE_PARAM_SAMPLE_COUNTS.items():
        print(f"  {name}: n = {n}")

    print("\nMedianas y desviaciones estándar calculadas:")
    for name in FREE_PARAM_NAMES:
        print(
            f"  {name}: "
            f"mediana = {FREE_PARAM_MEDIAN[name]:.8f}, "
            f"std = {FREE_PARAM_STD[name]:.8f}"
        )

    selected_info = choose_datasets_by_ids(DATASETS_INFO, VALIDATION_DATASET_IDS)

    datasets = []
    print("\nCargando datasets:")
    for item in selected_info:
        print(f"  {item['id']:02d} - {item['name']}")
        datasets.append(build_dataset(item))

    results = []

    for dataset in datasets:
        print(f"\nCalculando muestras para dataset {dataset['id']:02d} - {dataset['name']}...")

        res = run_uncertainty_simulations(
            dataset,
            N_MONTE_CARLO,
            n_workers=N_MONTE_CARLO_WORKERS
        )

        results.append(res)

        sugar_metrics = res["sugar_metrics"]
        ethanol_metrics = res["ethanol_metrics"]
        costs = res["validation_costs"]

        print(f"  Simulaciones válidas: {res['n_valid_runs']}/{N_MONTE_CARLO}")
        print(f"  Costo total validación: {costs['validation_cost_total']:.6f}")
        print(
            f"  Azúcar -> RMSE: {sugar_metrics['rmse']:.4f}, "
            f"NRMSE: {100 * sugar_metrics['nrmse']:.2f}%, "
            f"cobertura min-max: {sugar_metrics['coverage']:.1f}%"
        )
        print(
            f"  Etanol -> error abs.: {ethanol_metrics['abs_error']:.4f}, "
            f"error rel.: {100 * ethanol_metrics['relative_error']:.2f}%"
        )

    plot_sugar_results(datasets, results)
    plot_ethanol_results(datasets, results)


if __name__ == "__main__":
    main()
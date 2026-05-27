"""
zenteno_mc_halo_bands.py

Validación con incertidumbre Monte Carlo para el modelo original de Zenteno:
1) Azúcares totales S = G + F
2) Etanol E

Cada figura tiene 4 subplots, uno por cada dataset de validación.

- Curva negra: simulación con mediana de los parámetros.
- Todas las simulaciones Monte Carlo: líneas tenues superpuestas.
- Halo rojo tenue: banda alrededor de cada simulación Monte Carlo; las zonas con mayor superposición se ven más intensas.
- Puntos azules: datos experimentales.

IMPORTANTE:
- Este script asume el modelo original de Zenteno con estados [X, N, G, F, E].
- Usa los archivos, imports y parámetros del script original de Zenteno.
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
CANDIDATE_DIRS = [
    CURRENT_DIR,
    os.path.abspath(os.path.join(CURRENT_DIR, "..")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")),
]
for d in CANDIDATE_DIRS:
    if d not in sys.path:
        sys.path.insert(0, d)

from simulacion import data_for_simulation, simulate_system
from pymoo_opt import PARAM_ORDER

try:
    from pymoo_opt import compute_objective_breakdown
except ImportError:
    compute_objective_breakdown = None


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VALIDATION_DATASET_IDS = [3, 4, 11, 14]
DATASET_MARKERS = ["o", "s", "^", "*"]

N_MONTE_CARLO = 100
N_MONTE_CARLO_WORKERS = 4
MC_STD_WINDOW = 2  # 1 o 2 desviaciones estándar alrededor de la mediana
RANDOM_SEED = 123
TITLE_WRAP_WIDTH = 42
PENALTY_COST = 1e12


# ============================================================
# BOUNDS DE PARÁMETROS - ZENTENO ORIGINAL
# ============================================================

BOUNDS_DICT = {
    "mu0": (1e-2, 0.8),
    "betaG0": (1e-2, 10.0),
    "betaF0": (1e-2, 10.0),
    "Kn0": (1e-3, 1.0),
    "Kg0": (1e-1, 100.0),
    "Kf0": (1e-1, 100.0),
    "Kig0": (1e-1, 100.0),
    "Kie0": (1e-1, 100.0),
    "Kd0": (1e-5, 1.0),
    "Yxn": (1e-1, 10.0),
    "Yxg": (1e-2, 1.0),
    "Yxf": (1e-2, 10.0),
    "Yeg": (1e-1, 1.0),
    "Yef": (1e-1, 1.0),
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
# PARÁMETROS ESTIMADOS Y FIJOS - 38 SOLUCIONES
# ============================================================

FREE_PARAM_SAMPLES = {
    "mu0": [
        0.15341142, 0.04971307, 0.10799311, 0.10380503, 0.13171107,
        0.04461172, 0.06472872, 0.07959167, 0.16914052, 0.04647464,
        0.10337587, 0.07633801, 0.08748150, 0.06967066, 0.08803287,
        0.07488790, 0.05210100, 0.09876637, 0.10091901, 0.13597567,
        0.10078062, 0.14191535, 0.10944646, 0.04964366, 0.13037461,
        0.09537173, 0.09528157, 0.08563119, 0.13543102, 0.12474098,
        0.08946712, 0.12482276, 0.14829744, 0.14069450, 0.12270945,
        0.10919432, 0.11696696, 0.13621594,
    ],
    "betaG0": [
        0.95909898, 1.72828908, 1.27962137, 2.15101997, 1.28037803,
        1.28600071, 3.97020940, 1.27758967, 3.54050267, 3.76059626,
        1.58135086, 2.14843203, 1.44617274, 3.10600148, 1.65971105,
        3.01552241, 1.12749390, 3.64654027, 1.20772442, 3.10138398,
        2.73995511, 3.07890135, 2.09119385, 2.97227454, 2.99909832,
        3.33122621, 4.03776096, 2.16757455, 3.94839384, 1.46470985,
        2.06176677, 1.66130229, 3.79121878, 4.37514056, 2.82019047,
        3.17467257, 2.80735413, 2.27101677,
    ],
    "betaF0": [
        3.17393558, 0.31311858, 4.29937268, 4.20884849, 4.64855385,
        2.71896842, 0.78540917, 4.02670910, 1.18053864, 2.29573348,
        2.36568526, 1.31681044, 1.47542910, 5.34142393, 4.75321431,
        0.33911376, 3.09204250, 4.10614647, 2.74926546, 0.38802769,
        0.41024013, 4.37288291, 5.42971468, 0.29185520, 4.89363392,
        0.29589318, 5.10831279, 0.44371357, 0.38672266, 4.56051092,
        2.80644922, 0.27074728, 0.32192460, 0.35996107, 0.42149702,
        0.57718208, 4.42548634, 4.00595840,
    ],
    "Yxn": [
        4.49494067, 1.91080341, 2.10063375, 1.68702761, 2.51929876,
        3.08878053, 2.09311512, 1.77793115, 2.16077351, 1.42837967,
        3.85316296, 2.99588579, 2.50736216, 3.27060600, 3.76187587,
        3.75420864, 1.28197442, 3.41843278, 1.87863936, 1.19064842,
        2.47123037, 1.18160097, 1.16604141, 2.93933552, 1.44210034,
        2.13896940, 3.96910478, 1.34285940, 2.65921324, 2.62951027,
        3.25562082, 2.42511540, 2.11150874, 2.55230965, 5.05692759,
        4.36350493, 2.87835164, 4.25232543,
    ],
    "Yxg": [
        0.83074838, 0.37203382, 0.17314389, 0.99588913, 0.98225824,
        0.34308950, 0.63577383, 0.35553289, 0.21868183, 0.82252606,
        0.55617112, 0.61806703, 0.86465861, 0.99439373, 0.94292400,
        0.82495782, 0.93070146, 0.81419279, 0.26201799, 0.88822287,
        0.90954388, 0.54244361, 0.97176326, 0.99855160, 0.80815536,
        0.99223059, 0.84916497, 0.99954290, 0.58836422, 0.99829187,
        0.72470652, 0.99638664, 0.65243316, 0.81762131, 0.99550952,
        0.66895494, 0.99944616, 0.95427111,
    ],
    "Yeg": [
        0.35680740, 0.532098668, 0.527266568, 0.507892626, 0.603498841,
        0.57997025, 0.546017839, 0.392043895, 0.548911504, 0.568216264,
        0.39369724, 0.600281496, 0.347884189, 0.450990995, 0.598714159,
        0.404692484, 0.29257548, 0.598521587, 0.461673427, 0.343779878,
        0.559171174, 0.529731557, 0.612399081, 0.412607146, 0.613209623,
        0.375887613, 0.466118576, 0.61622095, 0.404111221, 0.625542333,
        0.552320368, 0.457347008, 0.354095728, 0.377290423, 0.538101475,
        0.59821293, 0.36345242, 0.543842414,
    ],
    "Yef": [
        0.452670279, 0.39887351, 0.246699956, 0.343058046, 0.244315537,
        0.343933172, 0.330463167, 0.350950672, 0.438802774, 0.236986123,
        0.320187146, 0.314949455, 0.477252949, 0.35570151, 0.436216172,
        0.163838146, 0.371321481, 0.40732258, 0.33301313, 0.173054696,
        0.4600088, 0.51948429, 0.381873949, 0.381533311, 0.468726654,
        0.415681295, 0.42550928, 0.385304086, 0.139422728, 0.443398097,
        0.329303541, 0.1609931, 0.300971475, 0.457138893, 0.217952642,
        0.364510076, 0.421122221, 0.336421008,
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
    free_param_median = {}
    free_param_std = {}

    for name, values in free_param_samples.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.ndim != 1:
            raise ValueError(f"Los valores de '{name}' deben ser una lista 1D.")
        if len(arr) < 2:
            raise ValueError(f"'{name}' debe tener al menos 2 valores válidos.")
        if name not in BOUNDS_DICT:
            raise ValueError(f"El parámetro libre '{name}' no está en BOUNDS_DICT.")

        free_param_median[name] = float(np.nanmedian(arr))
        free_param_std[name] = float(np.nanstd(arr, ddof=1))

    return free_param_median, free_param_std


def build_free_param_matrix(free_param_samples):
    free_names = list(free_param_samples.keys())
    lengths = {name: len(free_param_samples[name]) for name in free_names}
    max_len = max(lengths.values())
    matrix = np.full((max_len, len(free_names)), np.nan, dtype=float)

    for j, name in enumerate(free_names):
        arr = np.asarray(free_param_samples[name], dtype=float)
        matrix[:len(arr), j] = arr

    return free_names, matrix


def get_param_sample_counts(free_param_samples):
    return {name: len(values) for name, values in free_param_samples.items()}


def format_param_sample_counts(sample_counts):
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
        raise ValueError(f"Faltan parámetros según PARAM_ORDER: {missing_params}")
    return params


def sample_truncated_normal_parameter_with_window(name, rng, sigma_window):
    median_value = FREE_PARAM_MEDIAN[name]
    std_value = FREE_PARAM_STD[name]
    lb, ub = BOUNDS_DICT[name]
    median_value = float(np.clip(median_value, lb, ub))

    if sigma_window not in (1, 2):
        raise ValueError("MC_STD_WINDOW debe ser 1 o 2.")
    if not np.isfinite(std_value) or std_value <= 0:
        return median_value

    local_lb = max(lb, median_value - sigma_window * std_value)
    local_ub = min(ub, median_value + sigma_window * std_value)
    if local_ub <= local_lb:
        return median_value

    a = (local_lb - median_value) / std_value
    b = (local_ub - median_value) / std_value
    return float(truncnorm.rvs(a=a, b=b, loc=median_value, scale=std_value, random_state=rng))


def sample_free_params_truncnorm(seed=None, sigma_window=MC_STD_WINDOW):
    rng = np.random.default_rng(seed)
    return {
        name: sample_truncated_normal_parameter_with_window(name, rng, sigma_window)
        for name in FREE_PARAM_NAMES
    }


def build_sampled_param_dict(seed=None, sigma_window=MC_STD_WINDOW):
    params = FIXED_PARAMS.copy()
    params.update(sample_free_params_truncnorm(seed=seed, sigma_window=sigma_window))
    missing_params = [name for name in PARAM_ORDER if name not in params]
    if missing_params:
        raise ValueError(f"Faltan parámetros según PARAM_ORDER: {missing_params}")
    return params


def build_param_vector(param_dict):
    return np.array([param_dict[name] for name in PARAM_ORDER], dtype=float)


def build_median_theta_vector():
    return np.array([FREE_PARAM_MEDIAN[name] for name in FREE_PARAM_NAMES], dtype=float)


# ============================================================
# UTILIDADES DE DATASETS
# ============================================================

def choose_datasets_by_ids(datasets_info, dataset_ids):
    if len(dataset_ids) == 0:
        raise ValueError("Debes entregar al menos un ID de dataset.")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(f"Hay IDs repetidos en VALIDATION_DATASET_IDS: {dataset_ids}")

    dataset_map = {item["id"]: item for item in datasets_info}
    missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in dataset_map]
    if missing_ids:
        raise ValueError(f"IDs no encontrados en DATASETS_INFO: {missing_ids}")

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


def clean_dataset_name(name):
    return os.path.splitext(name)[0]


# ============================================================
# SIMULACIÓN
# ============================================================

def simulate_dataset(dataset, params_dict):
    """Simula Zenteno original: estados [X, N, G, F, E]."""
    params_vector = build_param_vector(params_dict)
    x0_og = np.asarray(dataset["x0"], dtype=float)

    X0 = x0_og[0]
    N0 = x0_og[1]
    E0 = x0_og[4] if len(x0_og) > 4 else 0.0

    S0 = float(dataset["sugar_initial"]) if dataset["sugar_initial"] is not None else 0.0
    G0 = S0 / 2.0
    F0 = S0 / 2.0
    x0 = np.array([X0, N0, G0, F0, E0], dtype=float)

    sol = simulate_system(
        x0=x0,
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector,
    )

    y = sol.y.T
    sugars = np.asarray(y[:, 2] + y[:, 3], dtype=float)
    ethanol = np.asarray(y[:, 4], dtype=float)

    if not np.all(np.isfinite(sugars)):
        raise RuntimeError("La simulación produjo valores no finitos en azúcares.")
    if not np.all(np.isfinite(ethanol)):
        raise RuntimeError("La simulación produjo valores no finitos en etanol.")

    return {"time": np.asarray(sol.t, dtype=float), "sugars": sugars, "ethanol": ethanol}


# ============================================================
# MÉTRICAS Y COSTOS
# ============================================================

def compute_sugar_validation_metrics(dataset, result):
    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)
    t_sim = np.asarray(result["time"], dtype=float)
    sugar_central = np.asarray(result["sugars_central"], dtype=float)
    sugar_interp = np.interp(t_exp, t_sim, sugar_central)

    valid = np.isfinite(t_exp) & np.isfinite(sugar_exp) & np.isfinite(sugar_interp)
    if not np.any(valid):
        return {"rmse": np.nan, "nrmse": np.nan, "coverage": np.nan, "n_exp_valid": 0}

    y_exp = sugar_exp[valid]
    y_sim = sugar_interp[valid]
    rmse = float(np.sqrt(np.mean((y_exp - y_sim) ** 2)))
    y_range = float(np.nanmax(y_exp) - np.nanmin(y_exp))
    nrmse = float(rmse / y_range) if y_range > 1e-8 else np.nan

    low_interp = np.interp(t_exp[valid], t_sim, result["sugar_min_max_bands"]["min"])
    high_interp = np.interp(t_exp[valid], t_sim, result["sugar_min_max_bands"]["max"])
    coverage = float(100.0 * np.mean((y_exp >= low_interp) & (y_exp <= high_interp)))

    return {"rmse": rmse, "nrmse": nrmse, "coverage": coverage, "n_exp_valid": int(np.sum(valid))}


def compute_ethanol_validation_metrics(dataset, result):
    et_exp = float(dataset["Et_final_exp"])
    et_central_final = float(result["ethanol_central"][-1])
    error = et_central_final - et_exp
    abs_error = abs(error)
    relative_error = abs_error / max(abs(et_exp), 1e-8)
    return {"error": error, "abs_error": abs_error, "relative_error": relative_error}


def compute_validation_costs(dataset):
    if compute_objective_breakdown is not None:
        try:
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
        except Exception as e:
            print(f"Advertencia: no se pudo usar compute_objective_breakdown ({e}). Se calculará costo manual.")

    sim = simulate_dataset(dataset, build_median_param_dict())
    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)
    sugar_sim = np.interp(t_exp, sim["time"], sim["sugars"])

    valid = np.isfinite(sugar_exp) & np.isfinite(sugar_sim)
    scale_s = max(float(np.nanmax(np.abs(sugar_exp[valid]))), 1e-8) if np.any(valid) else 1.0
    sugar_error = float(np.mean(((sugar_sim[valid] - sugar_exp[valid]) / scale_s) ** 2)) if np.any(valid) else np.nan

    et_exp = float(dataset["Et_final_exp"])
    ethanol_error = float(((sim["ethanol"][-1] - et_exp) / max(abs(et_exp), 1e-8)) ** 2)

    return {
        "validation_cost_total": sugar_error + ethanol_error,
        "validation_cost_sugar": sugar_error,
        "validation_cost_ethanol": ethanol_error,
    }


# ============================================================
# MONTE CARLO
# ============================================================

def run_single_monte_carlo_iteration(dataset, seed):
    try:
        sampled_params = build_sampled_param_dict(seed=seed)
        sim = simulate_dataset(dataset, sampled_params)
        return {"sugars": sim["sugars"], "ethanol": sim["ethanol"]}
    except Exception:
        return None


def compute_min_max_envelope(runs):
    runs = np.asarray(runs, dtype=float)
    return {"min": np.min(runs, axis=0), "max": np.max(runs, axis=0)}


def run_uncertainty_simulations(dataset, n_mc, n_workers=1):
    median_params = build_median_param_dict()
    central_sim = simulate_dataset(dataset, median_params)

    rng = np.random.default_rng(RANDOM_SEED + int(dataset["id"]) * 1000)
    seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=n_mc, dtype=np.uint32)

    sugar_runs = []
    ethanol_runs = []

    if n_workers == 1:
        for seed in seeds:
            mc_result = run_single_monte_carlo_iteration(dataset, int(seed))
            if mc_result is not None:
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
                if mc_result is not None:
                    sugar_runs.append(mc_result["sugars"])
                    ethanol_runs.append(mc_result["ethanol"])

    if len(sugar_runs) == 0:
        raise RuntimeError(f"No se pudieron generar simulaciones válidas para {dataset['name']}.")

    sugar_runs = np.asarray(sugar_runs, dtype=float)
    ethanol_runs = np.asarray(ethanol_runs, dtype=float)

    result = {
        "time": central_sim["time"],
        "sugars_central": central_sim["sugars"],
        "sugar_runs": sugar_runs,
        "sugar_min_max_bands": compute_min_max_envelope(sugar_runs),
        "ethanol_central": central_sim["ethanol"],
        "ethanol_runs": ethanol_runs,
        "ethanol_min_max_bands": compute_min_max_envelope(ethanol_runs),
        "n_valid_runs": len(sugar_runs),
    }

    result["validation_costs"] = compute_validation_costs(dataset)
    result["sugar_metrics"] = compute_sugar_validation_metrics(dataset, result)
    result["ethanol_metrics"] = compute_ethanol_validation_metrics(dataset, result)
    return result


# ============================================================
# PLOTS
# ============================================================

# ------------------------------------------------------------
# Configuración visual del halo Monte Carlo
# ------------------------------------------------------------
# La idea NO es construir bandas por percentiles.
# En cambio, cada trayectoria Monte Carlo se dibuja con:
#   1) una banda muy tenue alrededor de la línea, y
#   2) la línea Monte Carlo encima.
# Como todas las bandas se superponen, la zona por donde pasan más
# simulaciones se oscurece naturalmente.

MC_LINE_COLOR = "#c04a4a"
MC_LINE_ALPHA = 0.15
MC_LINE_WIDTH = 0.90

MC_HALO_COLOR = "#ff4d4d"
MC_HALO_ALPHA = 0.06
MC_HALO_WIDTH_FRACTION = 0.05
MC_HALO_MIN_WIDTH = 1e-6


def compute_halo_half_width(runs, width_fraction=MC_HALO_WIDTH_FRACTION):
    """
    Calcula el semi-ancho vertical del halo para las trayectorias MC.

    No usa percentiles. Solo toma el rango global de las simulaciones y
    define una fracción pequeña de ese rango como grosor visual del halo.
    """

    runs = np.asarray(runs, dtype=float)
    finite_values = runs[np.isfinite(runs)]

    if finite_values.size == 0:
        return MC_HALO_MIN_WIDTH

    y_range = float(np.nanmax(finite_values) - np.nanmin(finite_values))

    if not np.isfinite(y_range) or y_range <= 0:
        typical_scale = max(abs(float(np.nanmean(finite_values))), 1.0)
        return max(width_fraction * typical_scale, MC_HALO_MIN_WIDTH)

    return max(width_fraction * y_range, MC_HALO_MIN_WIDTH)


def plot_mc_runs_with_halo(
    ax,
    t_days,
    runs,
    color=MC_LINE_COLOR,
    halo_color=MC_HALO_COLOR,
    label_first=True,
):
    """
    Dibuja las simulaciones Monte Carlo con un halo tenue alrededor.

    La acumulación visual aparece porque los halos de muchas trayectorias
    se superponen: donde pasan más curvas, el rojo se ve más intenso.
    """

    runs = np.asarray(runs, dtype=float)
    halo_half_width = compute_halo_half_width(runs)

    for i, run in enumerate(runs):
        run = np.asarray(run, dtype=float)

        ax.fill_between(
            t_days,
            run - halo_half_width,
            run + halo_half_width,
            color=halo_color,
            alpha=MC_HALO_ALPHA,
            linewidth=0,
            label="Halo de simulaciones Monte Carlo" if (label_first and i == 0) else None,
            zorder=0,
        )

        ax.plot(
            t_days,
            run,
            color=color,
            alpha=MC_LINE_ALPHA,
            linewidth=MC_LINE_WIDTH,
            label="Simulaciones Monte Carlo" if (label_first and i == 0) else None,
            zorder=1,
        )


def create_2x2_axes(figsize=(16, 10.8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    return fig, axes.flatten()


def plot_sugar_results(datasets, results):
    fig, axes = create_2x2_axes()

    for idx, (ax, dataset, res) in enumerate(zip(axes, datasets, results)):
        t_sim_days = np.asarray(res["time"], dtype=float) / 24.0
        t_exp_days = np.asarray(dataset["t_rel"], dtype=float) / 24.0
        sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)
        valid_exp = np.isfinite(t_exp_days) & np.isfinite(sugar_exp)

        plot_mc_runs_with_halo(ax, t_sim_days, res["sugar_runs"], label_first=(idx == 0))

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

        ax.text(0.04, 0.55, text_box, transform=ax.transAxes, fontsize=10.0,
                va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))

        ax.set_title(
            f"Set {dataset['id']:02d} - {textwrap.fill(clean_dataset_name(dataset['name']), width=TITLE_WRAP_WIDTH)}",
            # f"Costo validación total: {costs['validation_cost_total']:.6f}",
            fontsize=10,
            pad=12,
        )
        ax.set_xlabel("Tiempo (días)", labelpad=8)
        ax.set_ylabel("Azúcares, S = G + F (g/L)", labelpad=8)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(
        "Validación predictiva del consumo de azúcares - modelo Zenteno\n"
        "Curva central con mediana de parámetros; 38 datos por parámetro\n"
        f"Halo con {N_MONTE_CARLO} muestras Monte Carlo "
        f"(muestreo truncado en mediana ±{MC_STD_WINDOW}σ)",
        fontsize=12,
        y=0.985,
    )
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.905), fontsize=9.5, frameon=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.80, hspace=0.4, wspace=0.194)
    plt.show()


def plot_ethanol_results(datasets, results):
    fig, axes = create_2x2_axes()

    for idx, (ax, dataset, res) in enumerate(zip(axes, datasets, results)):
        t_sim_days = np.asarray(res["time"], dtype=float) / 24.0
        et_exp = float(dataset["Et_final_exp"])
        t_final_exp_days = float(np.nanmax(np.asarray(dataset["t_rel"], dtype=float))) / 24.0

        plot_mc_runs_with_halo(ax, t_sim_days, res["ethanol_runs"], label_first=(idx == 0))

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

        ax.text(0.04, 0.94, text_box, transform=ax.transAxes, fontsize=10.0,
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
        "Validación predictiva de etanol final - modelo Zenteno\n"
        "Curva central con mediana de parámetros; 38 datos por parámetro\n"
        f"Halo con {N_MONTE_CARLO} muestras Monte Carlo "
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
    print("VALIDACIÓN CON BANDAS MONTE CARLO - ZENTENO ORIGINAL")
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
        print(f"  {name}: mediana = {FREE_PARAM_MEDIAN[name]:.8f}, std = {FREE_PARAM_STD[name]:.8f}")

    selected_info = choose_datasets_by_ids(DATASETS_INFO, VALIDATION_DATASET_IDS)

    datasets = []
    print("\nCargando datasets:")
    for item in selected_info:
        print(f"  {item['id']:02d} - {item['name']}")
        datasets.append(build_dataset(item))

    results = []
    for dataset in datasets:
        print(f"\nCalculando muestras para dataset {dataset['id']:02d} - {dataset['name']}...")
        res = run_uncertainty_simulations(dataset, N_MONTE_CARLO, n_workers=N_MONTE_CARLO_WORKERS)
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

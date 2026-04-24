# -*- coding: utf-8 -*-
"""
Gráficos de incertidumbre para datasets de validación elegidos manualmente
- Los datasets a graficar se indican explícitamente por ID
- Se simula con el promedio de parámetros
- Se construyen bandas con 100 simulaciones muestreadas
- Se grafica azúcares (G+F) y etanol en un mismo eje
- Figura final en formato 2x2
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation, simulate_system
from pymoo_opt import PARAM_ORDER, compute_objective_breakdown


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VALIDATION_DATASET_IDS = [3, 4, 11, 14]

N_MONTE_CARLO = 100
N_MONTE_CARLO_WORKERS = 4   # Si quieres serial, usa 1

LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95


# ============================================================
# LISTA DE LOS 16 DATASETS 100.000 L
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
# PARÁMETROS ESTIMADOS Y FIJOS
# ============================================================

# Con los 50 datos
FREE_PARAM_SAMPLES = {
    "mu0": [
        0.491229299, 0.095340553, 0.204922232, 0.238672303, 0.221547458,
        0.07104641, 0.12976963, 0.142978762, 0.231232081, 0.067254875,
        0.354456824, 0.1937975, 0.475079259, 0.140094304, 0.730494468,
        0.107069743, 0.139436427, 0.195722029, 0.244354817, 0.573844871,
        0.184323781, 0.135908409, 0.65230297, 0.132546263, 0.194138162,
        0.378589292, 0.175524911, 0.13869564, 0.052075002, 0.401838843,
        0.095131473, 0.32234151, 0.659623274, 0.059172439, 0.114620491,
        0.673564628, 0.10043929, 0.174521479, 0.109659677, 0.746238675,
        0.110052645, 0.126059526, 0.078883809, 0.590272845, 0.576184099,
        0.104120613, 0.153257031, 0.251328635, 0.140461114, 0.189262873
    ],
    "betaG0": [
        1.527080488, 4.850514939, 0.235535434, 1.478940534, 1.76326327,
        3.373843967, 1.100316472, 0.451875253, 1.718642886, 1.080347717,
        2.895646997, 1.305003993, 1.794283092, 2.576571471, 0.455945406,
        0.841086908, 1.437590477, 1.841314976, 1.688151119, 0.857812853,
        2.414963856, 3.303807082, 2.077030753, 3.893671185, 3.844084631,
        2.614446602, 4.157755196, 0.350965451, 1.813710079, 0.614042328,
        1.351497672, 1.999131619, 4.44825999, 1.118793644, 1.28223395,
        0.573888864, 4.380033293, 2.675765457, 2.955778978, 0.383232337,
        4.425205877, 0.808980602, 1.989834202, 0.729672539, 1.766676532,
        0.699872842, 4.285815386, 1.865459686, 3.644276357, 2.11834583
    ],
    "betaF0": [
        1.92590658, 1.723227134, 8.543200135, 1.137318706, 3.882215806,
        2.332203388, 3.276464489, 2.904638622, 2.997279946, 6.154276526,
        2.441717888, 2.162669254, 1.536080827, 2.857324395, 9.888852798,
        1.393066348, 1.175148555, 1.142151868, 1.102068367, 9.67621413,
        1.364981481, 6.990838079, 0.78075657, 0.672554907, 1.236836131,
        7.18444911, 1.475388007, 5.630079248, 1.566698496, 2.446273512,
        1.378767963, 1.405566722, 5.009808113, 8.136347988, 3.399773246,
        3.617017167, 0.635675162, 0.817523635, 2.187237852, 5.189603953,
        0.283873011, 1.650555251, 3.112934964, 5.200407589, 1.797508526,
        5.284849638, 2.536698845, 2.112764246, 0.010005176, 0.720895014
    ],
    "Yxn": [
        3.310792386, 2.416260149, 2.96828509, 4.748723377, 2.53667403,
        2.176098557, 2.197292445, 8.088028897, 2.354199463, 5.954931529,
        2.072011947, 3.533379451, 3.063003349, 1.945967411, 2.346502237,
        7.079418113, 6.527551915, 4.357026494, 5.073242529, 2.02325535,
        3.424252069, 0.659550545, 3.926339449, 3.921038779, 2.427412431,
        1.212010681, 2.067521245, 2.484893202, 5.864961797, 3.354371239,
        5.70467776, 3.56606508, 0.852325571, 7.346844279, 2.119873733,
        2.336016883, 2.951205022, 4.612656151, 2.655314944, 4.115073291,
        3.181449095, 6.346458486, 2.461554532, 1.992707409, 2.590517803,
        4.555679189, 1.731053717, 3.834171821, 9.99884258, 4.873346273
    ],
    "Yxg": [
        6.159759925, 9.991344746, 8.643858849, 5.580413761, 7.027678957,
        7.361149947, 5.625461807, 0.100248704, 8.235517554, 6.966119898,
        8.140244314, 9.958499624, 9.871331923, 9.98318681, 1.27459639,
        5.10248585, 6.234102728, 9.522424884, 9.457671514, 9.705414725,
        9.999774574, 6.617629792, 7.04277773, 9.991795607, 9.960162077,
        9.99304943, 9.996226996, 4.485346855, 4.20561476, 3.213024418,
        6.995949687, 6.164420288, 9.996898628, 1.537417306, 5.667888267,
        3.703244147, 9.909192638, 9.998376888, 9.000361131, 2.768893329,
        6.395466734, 7.207737619, 5.914804949, 3.781537417, 1.625587102,
        9.595856869, 5.697474361, 9.999686985, 8.389758258, 9.997187093
    ],
    "Yeg": [
        0.310224655, 0.580738471, 0.125913653, 0.358216272, 0.589804143,
        0.422721152, 0.223618323, 0.588613423, 0.334023374, 0.617444674,
        0.436034377, 0.274905805, 0.454176146, 0.342181722, 0.24741693,
        0.684141375, 0.320149437, 0.500510895, 0.566511753, 0.388004316,
        0.451717108, 0.269432437, 0.535934935, 0.570304104, 0.599681542,
        0.348454432, 0.564328667, 0.13702885, 0.199169038, 0.164883106,
        0.272824981, 0.496602885, 0.412136781, 0.729995622, 0.289485931,
        0.133601132, 0.602186605, 0.594369899, 0.393699066, 0.32734244,
        0.582809422, 0.606686477, 0.311984412, 0.209760328, 0.372835166,
        0.485217522, 0.653406549, 0.67902188, 0.720772616, 0.54004091
    ],
    "Yef": [
        0.546653089, 0.30876599, 0.597416543, 0.608281052, 0.299320903,
        0.412146877, 0.722679164, 0.441988632, 0.529533277, 0.319941909,
        0.473879284, 0.593714051, 0.370208632, 0.511170627, 0.600162515,
        0.221356485, 0.547531004, 0.330054858, 0.269391646, 0.532910919,
        0.461103743, 0.639075928, 0.211397763, 0.275390951, 0.192598341,
        0.53600703, 0.208208705, 0.6156564, 0.795467102, 0.896194082,
        0.706479051, 0.355080876, 0.626596208, 0.254434962, 0.508933506,
        0.965744813, 0.146048629, 0.255788255, 0.471549057, 0.537626386,
        0.100000301, 0.253045962, 0.529204287, 0.641928988, 0.540505426,
        0.421995461, 0.223571637, 0.2256859, 0.100000103, 0.278345778
    ],
}

# DATOS ELIMINACIÓN DE PUNTOS 1.
# FREE_PARAM_SAMPLES = {
#     "mu0": [
#         0.491229299, 0.095340553, 0.238672303, 0.221547458, 0.07104641,
#         0.12976963, 0.231232081, 0.354456824, 0.1937975, 0.475079259,
#         0.140094304, 0.107069743, 0.139436427, 0.195722029, 0.244354817,
#         0.184323781, 0.135908409, 0.65230297, 0.132546263, 0.194138162,
#         0.378589292, 0.175524911, 0.13869564, 0.052075002, 0.095131473,
#         0.32234151, 0.659623274, 0.114620491, 0.10043929, 0.174521479,
#         0.109659677, 0.110052645, 0.126059526, 0.078883809, 0.590272845,
#         0.576184099, 0.104120613, 0.153257031, 0.251328635, 0.189262873
#     ],
#     "betaG0": [
#         1.527080488, 4.850514939, 1.478940534, 1.76326327, 3.373843967,
#         1.100316472, 1.718642886, 2.895646997, 1.305003993, 1.794283092,
#         2.576571471, 0.841086908, 1.437590477, 1.841314976, 1.688151119,
#         2.414963856, 3.303807082, 2.077030753, 3.893671185, 3.844084631,
#         2.614446602, 4.157755196, 0.350965451, 1.813710079, 1.351497672,
#         1.999131619, 4.44825999, 1.28223395, 4.380033293, 2.675765457,
#         2.955778978, 4.425205877, 0.808980602, 1.989834202, 0.729672539,
#         1.766676532, 0.699872842, 4.285815386, 1.865459686, 2.11834583
#     ],
#     "betaF0": [
#         1.92590658, 1.723227134, 1.137318706, 3.882215806, 2.332203388,
#         3.276464489, 2.997279946, 2.441717888, 2.162669254, 1.536080827,
#         2.857324395, 1.393066348, 1.175148555, 1.142151868, 1.102068367,
#         1.364981481, 6.990838079, 0.78075657, 0.672554907, 1.236836131,
#         7.18444911, 1.475388007, 5.630079248, 1.566698496, 1.378767963,
#         1.405566722, 5.009808113, 3.399773246, 0.635675162, 0.817523635,
#         2.187237852, 0.283873011, 1.650555251, 3.112934964, 5.200407589,
#         1.797508526, 5.284849638, 2.536698845, 2.112764246, 0.720895014
#     ],
#     "Yxn": [
#         3.310792386, 2.416260149, 4.748723377, 2.53667403, 2.176098557,
#         2.197292445, 2.354199463, 2.072011947, 3.533379451, 3.063003349,
#         1.945967411, 7.079418113, 6.527551915, 4.357026494, 5.073242529,
#         3.424252069, 0.659550545, 3.926339449, 3.921038779, 2.427412431,
#         1.212010681, 2.067521245, 2.484893202, 5.864961797, 5.70467776,
#         3.56606508, 0.852325571, 2.119873733, 2.951205022, 4.612656151,
#         2.655314944, 3.181449095, 6.346458486, 2.461554532, 1.992707409,
#         2.590517803, 4.555679189, 1.731053717, 3.834171821, 4.873346273
#     ],
#     "Yxg": [
#         6.159759925, 9.991344746, 5.580413761, 7.027678957, 7.361149947,
#         5.625461807, 8.235517554, 8.140244314, 9.958499624, 9.871331923,
#         9.98318681, 5.10248585, 6.234102728, 9.522424884, 9.457671514,
#         9.999774574, 6.617629792, 7.04277773, 9.991795607, 9.960162077,
#         9.99304943, 9.996226996, 4.485346855, 4.20561476, 6.995949687,
#         6.164420288, 9.996898628, 5.667888267, 9.909192638, 9.998376888,
#         9.000361131, 6.395466734, 7.207737619, 5.914804949, 3.781537417,
#         1.625587102, 9.595856869, 5.697474361, 9.999686985, 9.997187093
#     ],
#     "Yeg": [
#         0.310224655, 0.580738471, 0.358216272, 0.589804143, 0.422721152,
#         0.223618323, 0.334023374, 0.436034377, 0.274905805, 0.454176146,
#         0.342181722, 0.684141375, 0.320149437, 0.500510895, 0.566511753,
#         0.451717108, 0.269432437, 0.535934935, 0.570304104, 0.599681542,
#         0.348454432, 0.564328667, 0.13702885, 0.199169038, 0.272824981,
#         0.496602885, 0.412136781, 0.289485931, 0.602186605, 0.594369899,
#         0.393699066, 0.582809422, 0.606686477, 0.311984412, 0.209760328,
#         0.372835166, 0.485217522, 0.653406549, 0.67902188, 0.54004091
#     ],
#     "Yef": [
#         0.546653089, 0.30876599, 0.608281052, 0.299320903, 0.412146877,
#         0.722679164, 0.529533277, 0.473879284, 0.593714051, 0.370208632,
#         0.511170627, 0.221356485, 0.547531004, 0.330054858, 0.269391646,
#         0.461103743, 0.639075928, 0.211397763, 0.275390951, 0.192598341,
#         0.53600703, 0.208208705, 0.6156564, 0.795467102, 0.706479051,
#         0.355080876, 0.626596208, 0.508933506, 0.146048629, 0.255788255,
#         0.471549057, 0.100000301, 0.253045962, 0.529204287, 0.641928988,
#         0.540505426, 0.421995461, 0.223571637, 0.2256859, 0.278345778
#     ],
# }

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

def choose_datasets_by_ids(datasets_info, dataset_ids):
    if len(dataset_ids) == 0:
        raise ValueError("Debes entregar al menos un ID de dataset.")

    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(f"Hay IDs repetidos en VALIDATION_DATASET_IDS: {dataset_ids}")

    dataset_map = {item["id"]: item for item in datasets_info}

    missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in dataset_map]
    if missing_ids:
        raise ValueError(
            f"Los siguientes IDs no existen en DATASETS_INFO: {missing_ids}"
        )

    selected = [dataset_map[dataset_id] for dataset_id in dataset_ids]
    return selected


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


def compute_mean_validation_cost(dataset, params_dict):
    params_vector = build_param_vector(params_dict)
    params_for_objective = {name: float(value) for name, value in zip(PARAM_ORDER, params_vector)}

    breakdown = compute_objective_breakdown(
        theta=params_vector,
        free_names=PARAM_ORDER,
        fixed_params={},
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        t_span=dataset["t_span"],
        sugars_profile=dataset["sugars_profile"],
        Et_final_exp=dataset["Et_final_exp"],
    )

    return {
        "objective_total": float(breakdown["objective_total"]),
        "sugar_error_mean": float(breakdown["sugar_error_mean"]),
        "ethanol_error": float(breakdown["ethanol_error"]),
        "params": params_for_objective,
    }


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
    validation_cost = compute_mean_validation_cost(dataset, mean_params)

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
        "validation_cost": validation_cost,
    }


# ============================================================
# PLOT
# ============================================================

def plot_results(datasets, results):
    n_plots = len(datasets)

    if n_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=False, sharey=False)
        if n_plots == 1:
            axes = [axes]

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
        ax.text(
            0.02,
            0.98,
            f"Costo validación (params promedio): {res['validation_cost']['objective_total']:.6f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.7"),
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
    print("GRÁFICOS DE INCERTIDUMBRE - DATASETS DEFINIDOS MANUALMENTE")
    print("=" * 80)

    print("\nPromedios calculados:")
    for k, v in FREE_PARAM_MEAN.items():
        print(f"{k}: {v:.8f}")

    print("\nDesviaciones estándar calculadas:")
    for k, v in FREE_PARAM_STD.items():
        print(f"{k}: {v:.8f}")

    print("\nIDs solicitados para validación:")
    print(VALIDATION_DATASET_IDS)

    selected_info = choose_datasets_by_ids(
        DATASETS_INFO,
        VALIDATION_DATASET_IDS
    )

    print("\nDatasets elegidos:")
    for item in selected_info:
        print(f"  {item['id']:02d} - {item['name']}")

    datasets = []
    for item in selected_info:
        print(f"\nCargando dataset {item['id']:02d}...")
        datasets.append(build_dataset(item))

    results = []
    validation_costs = []
    for dataset in datasets:
        print(f"\nCorriendo simulaciones para set {dataset['id']:02d} - {dataset['name']}")
        res = run_uncertainty_simulations(
            dataset,
            N_MONTE_CARLO,
            n_workers=N_MONTE_CARLO_WORKERS
        )
        results.append(res)
        validation_cost = res["validation_cost"]["objective_total"]
        validation_costs.append(validation_cost)
        print(f"Costo validación (promedio) set {dataset['id']:02d}: {validation_cost:.6f}")

    print("\nResumen de costos de validación (promedio):")
    for dataset, cost in zip(datasets, validation_costs):
        print(f"  Set {dataset['id']:02d}: {cost:.6f}")
    print(f"Suma de los cuatro: {float(np.sum(validation_costs)):.6f}")

    plot_results(datasets, results)


if __name__ == "__main__":
    main()
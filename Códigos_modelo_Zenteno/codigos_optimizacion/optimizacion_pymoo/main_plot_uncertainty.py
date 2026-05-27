"""
main_plot_uncertainty.py

Validación con bandas Monte Carlo para:
1) Azúcares totales S = G + F
2) Etanol E

Cada figura tiene 4 subplots, uno por cada dataset de validación.

- Curva negra: simulación con mediana de los parámetros.
- Sombra roja en 3 niveles:
    - 5-95%   : banda externa
    - 20-80%  : banda media
    - 35-65%  : banda interna
- Puntos azules:
    - Azúcares: perfil experimental de azúcares.
    - Etanol: valor experimental final de etanol.
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
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation, simulate_system
from pymoo_opt import PARAM_ORDER, compute_objective_breakdown


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VALIDATION_DATASET_IDS = [3, 4, 11, 14]

N_MONTE_CARLO = 200
N_MONTE_CARLO_WORKERS = 4

RANDOM_SEED = 123
TITLE_WRAP_WIDTH = 42
PENALTY_COST = 1e12


# ============================================================
# BOUNDS DE PARÁMETROS
# ============================================================

BOUNDS_DICT = {
    "mu0": (1e-2, 1.0),
    "betaG0": (1e-2, 10.0),
    "betaF0": (1e-2, 10.0),
    "Kn0": (1e-3, 1.0),
    "Kg0": (1e-1, 100.0),
    "Kf0": (1e-1, 100.0),
    "Kig0": (1e-1, 100.0),
    "Kie0": (1e-1, 100.0),
    "Yxn": (1e-1, 10.0),
    "Yxg": (1e-1, 10.0),
    "Yxf": (1e-1, 10.0),
    "Yeg": (1e-1, 10.0),
    "Yef": (1e-1, 10.0),
}


# ============================================================
# LISTA DE DATASETS
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

# SIN ELIMINACIÓN

# FREE_PARAM_SAMPLES = {
#     "mu0": [
#         0.491229299, 0.095340553, 0.204922232, 0.238672303, 0.221547458,
#         0.07104641, 0.12976963, 0.142978762, 0.231232081, 0.067254875,
#         0.354456824, 0.1937975, 0.475079259, 0.140094304, 0.730494468,
#         0.107069743, 0.139436427, 0.195722029, 0.244354817, 0.573844871,
#         0.184323781, 0.135908409, 0.65230297, 0.132546263, 0.194138162,
#         0.378589292, 0.175524911, 0.13869564, 0.052075002, 0.401838843,
#         0.095131473, 0.32234151, 0.659623274, 0.059172439, 0.114620491,
#         0.673564628, 0.10043929, 0.174521479, 0.109659677, 0.746238675,
#         0.110052645, 0.126059526, 0.078883809, 0.590272845, 0.576184099,
#         0.104120613, 0.153257031, 0.251328635, 0.140461114, 0.189262873
#     ],
#     "betaG0": [
#         1.527080488, 4.850514939, 0.235535434, 1.478940534, 1.76326327,
#         3.373843967, 1.100316472, 0.451875253, 1.718642886, 1.080347717,
#         2.895646997, 1.305003993, 1.794283092, 2.576571471, 0.455945406,
#         0.841086908, 1.437590477, 1.841314976, 1.688151119, 0.857812853,
#         2.414963856, 3.303807082, 2.077030753, 3.893671185, 3.844084631,
#         2.614446602, 4.157755196, 0.350965451, 1.813710079, 0.614042328,
#         1.351497672, 1.999131619, 4.44825999, 1.118793644, 1.28223395,
#         0.573888864, 4.380033293, 2.675765457, 2.955778978, 0.383232337,
#         4.425205877, 0.808980602, 1.989834202, 0.729672539, 1.766676532,
#         0.699872842, 4.285815386, 1.865459686, 3.644276357, 2.11834583
#     ],
#     "betaF0": [
#         1.92590658, 1.723227134, 8.543200135, 1.137318706, 3.882215806,
#         2.332203388, 3.276464489, 2.904638622, 2.997279946, 6.154276526,
#         2.441717888, 2.162669254, 1.536080827, 2.857324395, 9.888852798,
#         1.393066348, 1.175148555, 1.142151868, 1.102068367, 9.67621413,
#         1.364981481, 6.990838079, 0.78075657, 0.672554907, 1.236836131,
#         7.18444911, 1.475388007, 5.630079248, 1.566698496, 2.446273512,
#         1.378767963, 1.405566722, 5.009808113, 8.136347988, 3.399773246,
#         3.617017167, 0.635675162, 0.817523635, 2.187237852, 5.189603953,
#         0.283873011, 1.650555251, 3.112934964, 5.200407589, 1.797508526,
#         5.284849638, 2.536698845, 2.112764246, 0.010005176, 0.720895014
#     ],
#     "Yxn": [
#         3.310792386, 2.416260149, 2.96828509, 4.748723377, 2.53667403,
#         2.176098557, 2.197292445, 8.088028897, 2.354199463, 5.954931529,
#         2.072011947, 3.533379451, 3.063003349, 1.945967411, 2.346502237,
#         7.079418113, 6.527551915, 4.357026494, 5.073242529, 2.02325535,
#         3.424252069, 0.659550545, 3.926339449, 3.921038779, 2.427412431,
#         1.212010681, 2.067521245, 2.484893202, 5.864961797, 3.354371239,
#         5.70467776, 3.56606508, 0.852325571, 7.346844279, 2.119873733,
#         2.336016883, 2.951205022, 4.612656151, 2.655314944, 4.115073291,
#         3.181449095, 6.346458486, 2.461554532, 1.992707409, 2.590517803,
#         4.555679189, 1.731053717, 3.834171821, 9.99884258, 4.873346273
#     ],
#     "Yxg": [
#         6.159759925, 9.991344746, 8.643858849, 5.580413761, 7.027678957,
#         7.361149947, 5.625461807, 0.100248704, 8.235517554, 6.966119898,
#         8.140244314, 9.958499624, 9.871331923, 9.98318681, 1.27459639,
#         5.10248585, 6.234102728, 9.522424884, 9.457671514, 9.705414725,
#         9.999774574, 6.617629792, 7.04277773, 9.991795607, 9.960162077,
#         9.99304943, 9.996226996, 4.485346855, 4.20561476, 3.213024418,
#         6.995949687, 6.164420288, 9.996898628, 1.537417306, 5.667888267,
#         3.703244147, 9.909192638, 9.998376888, 9.000361131, 2.768893329,
#         6.395466734, 7.207737619, 5.914804949, 3.781537417, 1.625587102,
#         9.595856869, 5.697474361, 9.999686985, 8.389758258, 9.997187093
#     ],
#     "Yeg": [
#         0.310224655, 0.580738471, 0.125913653, 0.358216272, 0.589804143,
#         0.422721152, 0.223618323, 0.588613423, 0.334023374, 0.617444674,
#         0.436034377, 0.274905805, 0.454176146, 0.342181722, 0.24741693,
#         0.684141375, 0.320149437, 0.500510895, 0.566511753, 0.388004316,
#         0.451717108, 0.269432437, 0.535934935, 0.570304104, 0.599681542,
#         0.348454432, 0.564328667, 0.13702885, 0.199169038, 0.164883106,
#         0.272824981, 0.496602885, 0.412136781, 0.729995622, 0.289485931,
#         0.133601132, 0.602186605, 0.594369899, 0.393699066, 0.32734244,
#         0.582809422, 0.606686477, 0.311984412, 0.209760328, 0.372835166,
#         0.485217522, 0.653406549, 0.67902188, 0.720772616, 0.54004091
#     ],
#     "Yef": [
#         0.546653089, 0.30876599, 0.597416543, 0.608281052, 0.299320903,
#         0.412146877, 0.722679164, 0.441988632, 0.529533277, 0.319941909,
#         0.473879284, 0.593714051, 0.370208632, 0.511170627, 0.600162515,
#         0.221356485, 0.547531004, 0.330054858, 0.269391646, 0.532910919,
#         0.461103743, 0.639075928, 0.211397763, 0.275390951, 0.192598341,
#         0.53600703, 0.208208705, 0.6156564, 0.795467102, 0.896194082,
#         0.706479051, 0.355080876, 0.626596208, 0.254434962, 0.508933506,
#         0.965744813, 0.146048629, 0.255788255, 0.471549057, 0.537626386,
#         0.100000301, 0.253045962, 0.529204287, 0.641928988, 0.540505426,
#         0.421995461, 0.223571637, 0.2256859, 0.100000103, 0.278345778
#     ],
# }

## POST ELIMINACIÓN 1
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

## POST ELIMINACIÓN 2
FREE_PARAM_SAMPLES = {
    "mu0": [
        0.491229299, 0.095340553, 0.238672303, 0.221547458, 0.07104641,
        0.12976963, 0.231232081, 0.354456824, 0.1937975, 0.475079259,
        0.140094304, 0.107069743, 0.139436427, 0.195722029, 0.244354817,
        0.184323781, 0.132546263, 0.194138162, 0.378589292, 0.175524911,
        0.13869564, 0.052075002, 0.095131473, 0.32234151, 0.114620491,
        0.10043929, 0.174521479, 0.109659677, 0.110052645, 0.126059526,
        0.078883809, 0.104120613, 0.153257031, 0.251328635, 0.189262873
    ],
    "betaG0": [
        1.527080488, 4.850514939, 1.478940534, 1.76326327, 3.373843967,
        1.100316472, 1.718642886, 2.895646997, 1.305003993, 1.794283092,
        2.576571471, 0.841086908, 1.437590477, 1.841314976, 1.688151119,
        2.414963856, 3.893671185, 3.844084631, 2.614446602, 4.157755196,
        0.350965451, 1.813710079, 1.351497672, 1.999131619, 1.28223395,
        4.380033293, 2.675765457, 2.955778978, 4.425205877, 0.808980602,
        1.989834202, 0.699872842, 4.285815386, 1.865459686, 2.11834583
    ],
    "betaF0": [
        1.92590658, 1.723227134, 1.137318706, 3.882215806, 2.332203388,
        3.276464489, 2.997279946, 2.441717888, 2.162669254, 1.536080827,
        2.857324395, 1.393066348, 1.175148555, 1.142151868, 1.102068367,
        1.364981481, 0.672554907, 1.236836131, 7.18444911, 1.475388007,
        5.630079248, 1.566698496, 1.378767963, 1.405566722, 3.399773246,
        0.635675162, 0.817523635, 2.187237852, 0.283873011, 1.650555251,
        3.112934964, 5.284849638, 2.536698845, 2.112764246, 0.720895014
    ],
    "Yxn": [
        3.310792386, 2.416260149, 4.748723377, 2.53667403, 2.176098557,
        2.197292445, 2.354199463, 2.072011947, 3.533379451, 3.063003349,
        1.945967411, 7.079418113, 6.527551915, 4.357026494, 5.073242529,
        3.424252069, 3.921038779, 2.427412431, 1.212010681, 2.067521245,
        2.484893202, 5.864961797, 5.70467776, 3.56606508, 2.119873733,
        2.951205022, 4.612656151, 2.655314944, 3.181449095, 6.346458486,
        2.461554532, 4.555679189, 1.731053717, 3.834171821, 4.873346273
    ],
    "Yxg": [
        6.159759925, 9.991344746, 5.580413761, 7.027678957, 7.361149947,
        5.625461807, 8.235517554, 8.140244314, 9.958499624, 9.871331923,
        9.98318681, 5.10248585, 6.234102728, 9.522424884, 9.457671514,
        9.999774574, 9.991795607, 9.960162077, 9.99304943, 9.996226996,
        4.485346855, 4.20561476, 6.995949687, 6.164420288, 5.667888267,
        9.909192638, 9.998376888, 9.000361131, 6.395466734, 7.207737619,
        5.914804949, 9.595856869, 5.697474361, 9.999686985, 9.997187093
    ],
    "Yeg": [
        0.310224655, 0.580738471, 0.358216272, 0.589804143, 0.422721152,
        0.223618323, 0.334023374, 0.436034377, 0.274905805, 0.454176146,
        0.342181722, 0.684141375, 0.320149437, 0.500510895, 0.566511753,
        0.451717108, 0.570304104, 0.599681542, 0.348454432, 0.564328667,
        0.13702885, 0.199169038, 0.272824981, 0.496602885, 0.289485931,
        0.602186605, 0.594369899, 0.393699066, 0.582809422, 0.606686477,
        0.311984412, 0.485217522, 0.653406549, 0.67902188, 0.54004091
    ],
    "Yef": [
        0.546653089, 0.30876599, 0.608281052, 0.299320903, 0.412146877,
        0.722679164, 0.529533277, 0.473879284, 0.593714051, 0.370208632,
        0.511170627, 0.221356485, 0.547531004, 0.330054858, 0.269391646,
        0.461103743, 0.275390951, 0.192598341, 0.53600703, 0.208208705,
        0.6156564, 0.795467102, 0.706479051, 0.355080876, 0.508933506,
        0.146048629, 0.255788255, 0.471549057, 0.100000301, 0.253045962,
        0.529204287, 0.421995461, 0.223571637, 0.2256859, 0.278345778
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

        if arr.ndim != 1:
            raise ValueError(f"Los valores de '{name}' deben ser una lista 1D.")

        if len(arr) < 2:
            raise ValueError(
                f"'{name}' debe tener al menos 2 valores para calcular desviación estándar."
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
    free_names = list(free_param_samples.keys())

    lengths = [len(free_param_samples[name]) for name in free_names]
    if len(set(lengths)) != 1:
        raise ValueError(
            "Todas las listas de parámetros libres deben tener el mismo largo. "
            f"Largos encontrados: {dict(zip(free_names, lengths))}"
        )

    matrix = np.column_stack([
        np.asarray(free_param_samples[name], dtype=float)
        for name in free_names
    ])

    return free_names, matrix


FREE_PARAM_MEDIAN, FREE_PARAM_STD = compute_free_param_statistics(FREE_PARAM_SAMPLES)
FREE_PARAM_NAMES, FREE_PARAM_MATRIX = build_free_param_matrix(FREE_PARAM_SAMPLES)


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
    median_value = FREE_PARAM_MEDIAN[name]
    std_value = FREE_PARAM_STD[name]
    lb, ub = BOUNDS_DICT[name]

    median_value = float(np.clip(median_value, lb, ub))

    if not np.isfinite(std_value) or std_value <= 0:
        return median_value

    a = (lb - median_value) / std_value
    b = (ub - median_value) / std_value

    sampled_value = truncnorm.rvs(
        a=a,
        b=b,
        loc=median_value,
        scale=std_value,
        random_state=rng
    )

    return float(sampled_value)


def sample_free_params_truncnorm(seed=None):
    rng = np.random.default_rng(seed)

    sampled = {}
    for name in FREE_PARAM_NAMES:
        sampled[name] = sample_truncated_normal_parameter(name, rng)

    return sampled


def build_sampled_param_dict(seed=None):
    params = FIXED_PARAMS.copy()
    params.update(sample_free_params_truncnorm(seed=seed))

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
    - azúcares totales S = G + F
    - etanol E
    """

    params_vector = build_param_vector(params_dict)
    x0_og = np.asarray(dataset["x0"], dtype=float)

    # Extraer estados desde el x0 antiguo
    X0 = x0_og[0]
    N0 = x0_og[1]
    E0 = x0_og[4]

    # Cambiar azúcares
    if dataset["sugar_initial"] is not None:
        S0 = float(dataset["sugar_initial"])
    else:
        S0 = 0.0

    G0 = S0 / 2
    F0 = S0 / 2

    x0 = np.array([X0, N0, G0, F0, E0], dtype=float)

    sol = simulate_system(
        x0=x0,
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector
    )

    y = sol.y.T

    sugars = np.asarray(y[:, 2] + y[:, 3], dtype=float)
    ethanol = np.asarray(y[:, 4], dtype=float)

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

    low_interp = np.interp(
        t_exp[valid],
        t_sim,
        result["sugar_percentile_bands"]["p05"]
    )
    high_interp = np.interp(
        t_exp[valid],
        t_sim,
        result["sugar_percentile_bands"]["p95"]
    )

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


def compute_percentile_bands(runs):
    """
    Calcula 3 bandas percentilares anidadas:

    Banda externa:
        p05 - p95

    Banda media:
        p20 - p80

    Banda interna:
        p35 - p65

    Además calcula p50 como mediana Monte Carlo.
    """

    runs = np.asarray(runs, dtype=float)

    return {
        "p05": np.percentile(runs, 5, axis=0),
        "p20": np.percentile(runs, 20, axis=0),
        "p35": np.percentile(runs, 35, axis=0),
        "p50": np.percentile(runs, 50, axis=0),
        "p65": np.percentile(runs, 65, axis=0),
        "p80": np.percentile(runs, 80, axis=0),
        "p95": np.percentile(runs, 95, axis=0),
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

    sugar_bands = compute_percentile_bands(sugar_runs)
    ethanol_bands = compute_percentile_bands(ethanol_runs)

    result = {
        "time": central_sim["time"],

        "sugars_central": central_sim["sugars"],
        "sugar_mc_median": sugar_bands["p50"],
        "sugar_percentile_bands": sugar_bands,

        "ethanol_central": central_sim["ethanol"],
        "ethanol_mc_median": ethanol_bands["p50"],
        "ethanol_percentile_bands": ethanol_bands,

        "n_valid_runs": len(sugar_runs),
    }

    result["validation_costs"] = compute_validation_costs(dataset)
    result["sugar_metrics"] = compute_sugar_validation_metrics(dataset, result)
    result["ethanol_metrics"] = compute_ethanol_validation_metrics(dataset, result)

    return result


# ============================================================
# PLOT AUXILIAR
# ============================================================

def add_nested_red_bands(ax, t, bands, label_first=True):
    """
    Agrega 3 bandas percentilares anidadas para representar densidad visual.

    Zonas:
    - 5-95%  : rango amplio, rojo claro.
    - 20-80% : rango intermedio, rojo medio.
    - 35-65% : zona más concentrada, rojo oscuro.
    """

    ax.fill_between(
        t,
        bands["p05"],
        bands["p95"],
        color="#f3a6a6",
        alpha=0.62,
        linewidth=0,
        label="MC 5-95%" if label_first else None
    )

    ax.fill_between(
        t,
        bands["p20"],
        bands["p80"],
        color="#e25f5f",
        alpha=0.52,
        linewidth=0,
        label="MC 20-80%" if label_first else None
    )

    ax.fill_between(
        t,
        bands["p35"],
        bands["p65"],
        color="#bf3737",
        alpha=0.48,
        linewidth=0,
        label="MC 35-65%" if label_first else None
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

def plot_sugar_results(datasets, results, n_param_samples):
    fig, axes = create_2x2_axes()

    for idx, (ax, dataset, res) in enumerate(zip(axes, datasets, results)):
        t_sim_days = np.asarray(res["time"], dtype=float) / 24.0
        t_exp_days = np.asarray(dataset["t_rel"], dtype=float) / 24.0

        sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)
        valid_exp = np.isfinite(t_exp_days) & np.isfinite(sugar_exp)

        add_nested_red_bands(
            ax,
            t_sim_days,
            res["sugar_percentile_bands"],
            label_first=(idx == 0)
        )

        ax.plot(
            t_sim_days,
            res["sugars_central"],
            color="black",
            linewidth=2.2,
            label="Simulación con la mediana de los parámetros"
        )

        ax.scatter(
            t_exp_days[valid_exp],
            sugar_exp[valid_exp],
            s=30,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="Azúcares experimentales"
        )

        metrics = res["sugar_metrics"]
        costs = res["validation_costs"]

        text_box = (
            f"RMSE: {metrics['rmse']:.2f} g/L\n"
            f"NRMSE: {100 * metrics['nrmse']:.2f}%\n"
            f"Costo azúcar: {costs['validation_cost_sugar']:.6f}\n"
        )

        ax.text(
            0.60,
            0.94,
            text_box,
            transform=ax.transAxes,
            fontsize=8.0,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88)
        )

        ax.set_title(
            f"{textwrap.fill(clean_dataset_name(dataset['name']), width=TITLE_WRAP_WIDTH)}\n"
            f"Costo validación total: {costs['validation_cost_total']:.6f}",
            fontsize=10,
            pad=12
        )

        ax.set_xlabel("Tiempo (días)", labelpad=8)
        ax.set_ylabel("Azúcares, S = G + F (g/L)", labelpad=8)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    fig.suptitle(
        "Validación predictiva del consumo de azúcares\n"
        f"Curva central con mediana de parámetros + bandas Monte Carlo con normal "
        f"({n_param_samples} muestras de parámetros)",
        fontsize=15,
        y=0.985
    )

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.905),
        fontsize=9.5,
        frameon=True
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.80,
        hspace=0.55,
        wspace=0.25
    )

    plt.show()


# ============================================================
# FIGURA 2: ETANOL
# ============================================================

def plot_ethanol_results(datasets, results, n_param_samples):
    fig, axes = create_2x2_axes()

    for idx, (ax, dataset, res) in enumerate(zip(axes, datasets, results)):
        t_sim_days = np.asarray(res["time"], dtype=float) / 24.0

        et_exp = float(dataset["Et_final_exp"])
        t_final_exp_days = float(np.nanmax(np.asarray(dataset["t_rel"], dtype=float))) / 24.0

        add_nested_red_bands(
            ax,
            t_sim_days,
            res["ethanol_percentile_bands"],
            label_first=(idx == 0)
        )

        ax.plot(
            t_sim_days,
            res["ethanol_central"],
            color="black",
            linewidth=2.2,
            label="Simulación con la mediana de los parámetros"
        )

        ax.scatter(
            [t_final_exp_days],
            [et_exp],
            s=55,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label="Etanol experimental final"
        )

        metrics = res["ethanol_metrics"]
        costs = res["validation_costs"]

        text_box = (
            f"Error abs.: {metrics['abs_error']:.2f} g/L\n"
            f"Error rel.: {100 * metrics['relative_error']:.2f}%\n"
            f"Costo etanol: {costs['validation_cost_ethanol']:.6f}"
        )

        ax.text(
            0.40,
            0.94,
            text_box,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88)
        )

        ax.set_title(
            f"{textwrap.fill(clean_dataset_name(dataset['name']), width=TITLE_WRAP_WIDTH)}\n"
            f"Costo validación total: {costs['validation_cost_total']:.6f}",
            fontsize=10,
            pad=12
        )

        ax.set_xlabel("Tiempo (días)", labelpad=8)
        ax.set_ylabel("Etanol, E (g/L)", labelpad=8)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    fig.suptitle(
        "Validación predictiva de etanol final\n"
        f"Curva central con mediana de parámetros + bandas Monte Carlo con normal "
        f"({n_param_samples} muestras de parámetros)",
        fontsize=15,
        y=0.985
    )

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.905),
        fontsize=9.5,
        frameon=True
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.80,
        hspace=0.55,
        wspace=0.25
    )

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

    n_samples = len(next(iter(FREE_PARAM_SAMPLES.values())))
    print(f"\nMuestras disponibles por parámetro libre: {n_samples}")

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
            # f"puntos en banda: {sugar_metrics['coverage']:.1f}%"
        )
        print(
            f"  Etanol -> error abs.: {ethanol_metrics['abs_error']:.4f}, "
            f"error rel.: {100 * ethanol_metrics['relative_error']:.2f}%"
        )

    plot_sugar_results(datasets, results, n_samples)
    plot_ethanol_results(datasets, results, n_samples)


if __name__ == "__main__":
    main()
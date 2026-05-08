"""
Validación de azúcares con bandas deterministas usando desviación estándar.

Este script reemplaza las simulaciones Monte Carlo por simulaciones puntuales:
- curva central: mediana de los parámetros libres
- banda k=1: mediana ± 1 desviación estándar en todos los parámetros libres
- banda k=2: mediana ± 2 desviaciones estándar en todos los parámetros libres

Se generan dos figuras:
1) Bandas con ±1 desviación estándar
2) Bandas con ±2 desviaciones estándar

En cada figura se superponen los 4 datasets de validación definidos en
VALIDATION_DATASET_IDS. Solo se grafica azúcar total S = G + F.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from simulacion_coleman import data_for_simulation, simulate_system
from pymoo_opt_coleman import PARAM_ORDER


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VALIDATION_DATASET_IDS = [3, 4, 11, 14]
TITLE_WRAP_WIDTH = 42

# Marcadores para distinguir datasets superpuestos
DATASET_MARKERS = ["o", "s", "^", "*"]


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
    "Kd0": (1e-4, 1e-1),
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

# 50 datos por parámetro libre (sin eliminar ningún dataset).
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

# Set post eliminación 2: 35 muestras por parámetro libre.
# FREE_PARAM_SAMPLES = {
#     "mu0": [
#         0.491229299, 0.095340553, 0.238672303, 0.221547458, 0.07104641,
#         0.12976963, 0.231232081, 0.354456824, 0.1937975, 0.475079259,
#         0.140094304, 0.107069743, 0.139436427, 0.195722029, 0.244354817,
#         0.184323781, 0.132546263, 0.194138162, 0.378589292, 0.175524911,
#         0.13869564, 0.052075002, 0.095131473, 0.32234151, 0.114620491,
#         0.10043929, 0.174521479, 0.109659677, 0.110052645, 0.126059526,
#         0.078883809, 0.104120613, 0.153257031, 0.251328635, 0.189262873
#     ],
#     "betaG0": [
#         1.527080488, 4.850514939, 1.478940534, 1.76326327, 3.373843967,
#         1.100316472, 1.718642886, 2.895646997, 1.305003993, 1.794283092,
#         2.576571471, 0.841086908, 1.437590477, 1.841314976, 1.688151119,
#         2.414963856, 3.893671185, 3.844084631, 2.614446602, 4.157755196,
#         0.350965451, 1.813710079, 1.351497672, 1.999131619, 1.28223395,
#         4.380033293, 2.675765457, 2.955778978, 4.425205877, 0.808980602,
#         1.989834202, 0.699872842, 4.285815386, 1.865459686, 2.11834583
#     ],
#     "betaF0": [
#         1.92590658, 1.723227134, 1.137318706, 3.882215806, 2.332203388,
#         3.276464489, 2.997279946, 2.441717888, 2.162669254, 1.536080827,
#         2.857324395, 1.393066348, 1.175148555, 1.142151868, 1.102068367,
#         1.364981481, 0.672554907, 1.236836131, 7.18444911, 1.475388007,
#         5.630079248, 1.566698496, 1.378767963, 1.405566722, 3.399773246,
#         0.635675162, 0.817523635, 2.187237852, 0.283873011, 1.650555251,
#         3.112934964, 5.284849638, 2.536698845, 2.112764246, 0.720895014
#     ],
#     "Yxn": [
#         3.310792386, 2.416260149, 4.748723377, 2.53667403, 2.176098557,
#         2.197292445, 2.354199463, 2.072011947, 3.533379451, 3.063003349,
#         1.945967411, 7.079418113, 6.527551915, 4.357026494, 5.073242529,
#         3.424252069, 3.921038779, 2.427412431, 1.212010681, 2.067521245,
#         2.484893202, 5.864961797, 5.70467776, 3.56606508, 2.119873733,
#         2.951205022, 4.612656151, 2.655314944, 3.181449095, 6.346458486,
#         2.461554532, 4.555679189, 1.731053717, 3.834171821, 4.873346273
#     ],
#     "Yxg": [
#         6.159759925, 9.991344746, 5.580413761, 7.027678957, 7.361149947,
#         5.625461807, 8.235517554, 8.140244314, 9.958499624, 9.871331923,
#         9.98318681, 5.10248585, 6.234102728, 9.522424884, 9.457671514,
#         9.999774574, 9.991795607, 9.960162077, 9.99304943, 9.996226996,
#         4.485346855, 4.20561476, 6.995949687, 6.164420288, 5.667888267,
#         9.909192638, 9.998376888, 9.000361131, 6.395466734, 7.207737619,
#         5.914804949, 9.595856869, 5.697474361, 9.999686985, 9.997187093
#     ],
#     "Yeg": [
#         0.310224655, 0.580738471, 0.358216272, 0.589804143, 0.422721152,
#         0.223618323, 0.334023374, 0.436034377, 0.274905805, 0.454176146,
#         0.342181722, 0.684141375, 0.320149437, 0.500510895, 0.566511753,
#         0.451717108, 0.570304104, 0.599681542, 0.348454432, 0.564328667,
#         0.13702885, 0.199169038, 0.272824981, 0.496602885, 0.289485931,
#         0.602186605, 0.594369899, 0.393699066, 0.582809422, 0.606686477,
#         0.311984412, 0.485217522, 0.653406549, 0.67902188, 0.54004091
#     ],
#     "Yef": [
#         0.546653089, 0.30876599, 0.608281052, 0.299320903, 0.412146877,
#         0.722679164, 0.529533277, 0.473879284, 0.593714051, 0.370208632,
#         0.511170627, 0.221356485, 0.547531004, 0.330054858, 0.269391646,
#         0.461103743, 0.275390951, 0.192598341, 0.53600703, 0.208208705,
#         0.6156564, 0.795467102, 0.706479051, 0.355080876, 0.508933506,
#         0.146048629, 0.255788255, 0.471549057, 0.100000301, 0.253045962,
#         0.529204287, 0.421995461, 0.223571637, 0.2256859, 0.278345778
#     ],
# }

# Set post eliminación 3: 30 muestras por parámetro libre.
# FREE_PARAM_SAMPLES = {
#     "mu0": [
#         0.095340553, 0.238672303, 0.221547458, 0.07104641, 0.12976963,
#         0.231232081, 0.354456824, 0.1937975, 0.475079259, 0.140094304,
#         0.139436427, 0.195722029, 0.244354817, 0.184323781, 0.132546263,
#         0.194138162, 0.175524911, 0.095131473, 0.32234151, 0.114620491,
#         0.10043929, 0.174521479, 0.109659677, 0.110052645, 0.126059526,
#         0.078883809, 0.104120613, 0.153257031, 0.251328635, 0.189262873
#     ],
#     "betaG0": [
#         4.850514939, 1.478940534, 1.76326327, 3.373843967, 1.100316472,
#         1.718642886, 2.895646997, 1.305003993, 1.794283092, 2.576571471,
#         1.437590477, 1.841314976, 1.688151119, 2.414963856, 3.893671185,
#         3.844084631, 4.157755196, 1.351497672, 1.999131619, 1.28223395,
#         4.380033293, 2.675765457, 2.955778978, 4.425205877, 0.808980602,
#         1.989834202, 0.699872842, 4.285815386, 1.865459686, 2.11834583
#     ],
#     "betaF0": [
#         1.723227134, 1.137318706, 3.882215806, 2.332203388, 3.276464489,
#         2.997279946, 2.441717888, 2.162669254, 1.536080827, 2.857324395,
#         1.175148555, 1.142151868, 1.102068367, 1.364981481, 0.672554907,
#         1.236836131, 1.475388007, 1.378767963, 1.405566722, 3.399773246,
#         0.635675162, 0.817523635, 2.187237852, 0.283873011, 1.650555251,
#         3.112934964, 5.284849638, 2.536698845, 2.112764246, 0.720895014
#     ],
#     "Kn0": [
#         0.009647, 0.009647, 0.009647, 0.009647, 0.009647,
#         0.009647, 0.009647, 0.009647, 0.009647, 0.009647,
#         0.009647, 0.009647, 0.009647, 0.009647, 0.009647,
#         0.009647, 0.009647, 0.009647, 0.009647, 0.009647,
#         0.009647, 0.009647, 0.009647, 0.009647, 0.009647,
#         0.009647, 0.009647, 0.009647, 0.009647, 0.009647
#     ],
#     "Kg0": [
#         8.551854, 8.551854, 8.551854, 8.551854, 8.551854,
#         8.551854, 8.551854, 8.551854, 8.551854, 8.551854,
#         8.551854, 8.551854, 8.551854, 8.551854, 8.551854,
#         8.551854, 8.551854, 8.551854, 8.551854, 8.551854,
#         8.551854, 8.551854, 8.551854, 8.551854, 8.551854,
#         8.551854, 8.551854, 8.551854, 8.551854, 8.551854
#     ],
#     "Kf0": [
#         7.16565, 7.16565, 7.16565, 7.16565, 7.16565,
#         7.16565, 7.16565, 7.16565, 7.16565, 7.16565,
#         7.16565, 7.16565, 7.16565, 7.16565, 7.16565,
#         7.16565, 7.16565, 7.16565, 7.16565, 7.16565,
#         7.16565, 7.16565, 7.16565, 7.16565, 7.16565,
#         7.16565, 7.16565, 7.16565, 7.16565, 7.16565
#     ],
#     "Kig0": [
#         44.15067, 44.15067, 44.15067, 44.15067, 44.15067,
#         44.15067, 44.15067, 44.15067, 44.15067, 44.15067,
#         44.15067, 44.15067, 44.15067, 44.15067, 44.15067,
#         44.15067, 44.15067, 44.15067, 44.15067, 44.15067,
#         44.15067, 44.15067, 44.15067, 44.15067, 44.15067,
#         44.15067, 44.15067, 44.15067, 44.15067, 44.15067
#     ],
#     "Kie0": [
#         42.528284, 42.528284, 42.528284, 42.528284, 42.528284,
#         42.528284, 42.528284, 42.528284, 42.528284, 42.528284,
#         42.528284, 42.528284, 42.528284, 42.528284, 42.528284,
#         42.528284, 42.528284, 42.528284, 42.528284, 42.528284,
#         42.528284, 42.528284, 42.528284, 42.528284, 42.528284,
#         42.528284, 42.528284, 42.528284, 42.528284, 42.528284
#     ],
#     "Kd0": [
#         0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
#         0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
#         0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
#         0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
#         0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
#         0.0001, 0.0001, 0.0001, 0.0001, 0.0001
#     ],
#     "Yxn": [
#         2.416260149, 4.748723377, 2.53667403, 2.176098557, 2.197292445,
#         2.354199463, 2.072011947, 3.533379451, 3.063003349, 1.945967411,
#         6.527551915, 4.357026494, 5.073242529, 3.424252069, 3.921038779,
#         2.427412431, 2.067521245, 5.70467776, 3.56606508, 2.119873733,
#         2.951205022, 4.612656151, 2.655314944, 3.181449095, 6.346458486,
#         2.461554532, 4.555679189, 1.731053717, 3.834171821, 4.873346273
#     ],
#     "Yxg": [
#         9.991344746, 5.580413761, 7.027678957, 7.361149947, 5.625461807,
#         8.235517554, 8.140244314, 9.958499624, 9.871331923, 9.98318681,
#         6.234102728, 9.522424884, 9.457671514, 9.999774574, 9.991795607,
#         9.960162077, 9.996226996, 6.995949687, 6.164420288, 5.667888267,
#         9.909192638, 9.998376888, 9.000361131, 6.395466734, 7.207737619,
#         5.914804949, 9.595856869, 5.697474361, 9.999686985, 9.997187093
#     ],
#     "Yxf": [
#         1.642634, 1.642634, 1.642634, 1.642634, 1.642634,
#         1.642634, 1.642634, 1.642634, 1.642634, 1.642634,
#         1.642634, 1.642634, 1.642634, 1.642634, 1.642634,
#         1.642634, 1.642634, 1.642634, 1.642634, 1.642634,
#         1.642634, 1.642634, 1.642634, 1.642634, 1.642634,
#         1.642634, 1.642634, 1.642634, 1.642634, 1.642634
#     ],
#     "Yeg": [
#         0.580738471, 0.358216272, 0.589804143, 0.422721152, 0.223618323,
#         0.334023374, 0.436034377, 0.274905805, 0.454176146, 0.342181722,
#         0.320149437, 0.500510895, 0.566511753, 0.451717108, 0.570304104,
#         0.599681542, 0.564328667, 0.272824981, 0.496602885, 0.289485931,
#         0.602186605, 0.594369899, 0.393699066, 0.582809422, 0.606686477,
#         0.311984412, 0.485217522, 0.653406549, 0.67902188, 0.54004091
#     ],
#     "Yef": [
#         0.30876599, 0.608281052, 0.299320903, 0.412146877, 0.722679164,
#         0.529533277, 0.473879284, 0.593714051, 0.370208632, 0.511170627,
#         0.547531004, 0.330054858, 0.269391646, 0.461103743, 0.275390951,
#         0.192598341, 0.208208705, 0.706479051, 0.355080876, 0.508933506,
#         0.146048629, 0.255788255, 0.471549057, 0.100000301, 0.253045962,
#         0.529204287, 0.421995461, 0.223571637, 0.2256859, 0.278345778
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
    """Calcula mediana y desviación estándar muestral para cada parámetro libre."""

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
            raise ValueError(f"El parámetro '{name}' no está en BOUNDS_DICT.")

        free_param_median[name] = float(np.nanmedian(arr))
        free_param_std[name] = float(np.nanstd(arr, ddof=1))

    return free_param_median, free_param_std


FREE_PARAM_NAMES = list(FREE_PARAM_SAMPLES.keys())
FREE_PARAM_MEDIAN, FREE_PARAM_STD = compute_free_param_statistics(FREE_PARAM_SAMPLES)
N_PARAM_SAMPLES = len(next(iter(FREE_PARAM_SAMPLES.values())))


def validate_free_param_sample_lengths():
    lengths = {name: len(values) for name, values in FREE_PARAM_SAMPLES.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Todas las listas de FREE_PARAM_SAMPLES deben tener el mismo largo. "
            f"Largos encontrados: {lengths}"
        )


def build_param_dict_from_free_values(free_values):
    """Combina parámetros fijos con valores dados para los parámetros libres."""

    params = FIXED_PARAMS.copy()
    params.update(free_values)

    missing_params = [name for name in PARAM_ORDER if name not in params]
    if missing_params:
        raise ValueError(
            "Faltan parámetros para construir el vector completo según PARAM_ORDER: "
            f"{missing_params}"
        )

    return params


def build_median_param_dict():
    return build_param_dict_from_free_values(FREE_PARAM_MEDIAN)


def build_std_shift_param_dict(k_std, sign):
    """
    Construye el vector de parámetros libres como:
        mediana + sign * k_std * desviación estándar

    Regla de corrección:
    - Solo si el valor calculado queda negativo, se reemplaza por el lower bound.
    - No se corrige por upper bound.
    """

    shifted_free_values = {}

    for name in FREE_PARAM_NAMES:
        median_value = FREE_PARAM_MEDIAN[name]
        std_value = FREE_PARAM_STD[name]
        lb, ub = BOUNDS_DICT[name]

        value = median_value + sign * k_std * std_value

        # Solo corregir si queda negativo
        if value < 0:
            value = lb

        shifted_free_values[name] = float(value)

    return build_param_dict_from_free_values(shifted_free_values)


def build_param_vector(param_dict):
    return np.array([param_dict[name] for name in PARAM_ORDER], dtype=float)


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


def clean_dataset_name(name):
    return name.replace(".xlsx", "")


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


# ============================================================
# SIMULACIÓN
# ============================================================

def simulate_dataset(dataset, params_dict):
    """Simula un dataset y retorna tiempo y azúcar total S = G + F."""

    params_vector = build_param_vector(params_dict)

    sol = simulate_system(
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector,
    )

    y = sol.y.T
    sugars = np.asarray(y[:, 2] + y[:, 3], dtype=float)

    if not np.all(np.isfinite(sugars)):
        raise RuntimeError("La simulación produjo valores no finitos en azúcares.")

    return {
        "time": np.asarray(sol.t, dtype=float),
        "sugars": sugars,
    }


def interpolate_to_experimental_times(simulation, t_exp):
    """Interpola la simulación a los mismos tiempos experimentales."""

    t_sim = np.asarray(simulation["time"], dtype=float)
    sugar_sim = np.asarray(simulation["sugars"], dtype=float)

    return np.interp(t_exp, t_sim, sugar_sim)


def simulate_dataset_with_std_band(dataset, k_std):
    """
    Para un dataset y un k_std, calcula:
    - curva central con mediana
    - curva con mediana + k_std*std
    - curva con mediana - k_std*std
    - banda low/high entre ambas curvas extremas

    Todo se devuelve evaluado en los tiempos experimentales.
    """

    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    sugar_exp = np.asarray(dataset["sugars_profile"], dtype=float)

    central_params = build_median_param_dict()
    plus_params = build_std_shift_param_dict(k_std=k_std, sign=1)
    minus_params = build_std_shift_param_dict(k_std=k_std, sign=-1)

    central_sim = simulate_dataset(dataset, central_params)
    plus_sim = simulate_dataset(dataset, plus_params)
    minus_sim = simulate_dataset(dataset, minus_params)

    sugar_central = interpolate_to_experimental_times(central_sim, t_exp)
    sugar_plus = interpolate_to_experimental_times(plus_sim, t_exp)
    sugar_minus = interpolate_to_experimental_times(minus_sim, t_exp)

    band_low = np.minimum(sugar_plus, sugar_minus)
    band_high = np.maximum(sugar_plus, sugar_minus)

    valid = np.isfinite(t_exp) & np.isfinite(sugar_exp) & np.isfinite(sugar_central)

    if np.any(valid):
        errors = sugar_exp[valid] - sugar_central[valid]
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        y_range = float(np.nanmax(sugar_exp[valid]) - np.nanmin(sugar_exp[valid]))
        nrmse = float(rmse / y_range) if y_range > 1e-8 else np.nan
    else:
        rmse = np.nan
        nrmse = np.nan

    return {
        "t_exp": t_exp,
        "sugar_exp": sugar_exp,
        "sugar_central": sugar_central,
        "band_low": band_low,
        "band_high": band_high,
        "valid": valid,
        "rmse": rmse,
        "nrmse": nrmse,
    }


# ============================================================
# GRÁFICOS
# ============================================================

def plot_std_band_figure(datasets, results_by_dataset, k_std):
    fig, ax = plt.subplots(figsize=(14, 8.5))

    for idx, (dataset, result) in enumerate(zip(datasets, results_by_dataset)):
        marker = DATASET_MARKERS[idx % len(DATASET_MARKERS)]
        t = result["t_exp"]
        valid = result["valid"]

        dataset_label = f"Set {dataset['id']}"

        # Banda roja suave de incertidumbre determinista.
        ax.fill_between(
            t[valid]/24,
            result["band_low"][valid],
            result["band_high"][valid],
            color="#f08080",
            alpha=0.18,
            linewidth=0,
            label=f"Banda ±{k_std} DE" if idx == 0 else None,
        )

        # Curva central con mediana de parámetros, evaluada solo en tiempos experimentales.
        ax.plot(
            t[valid]/24,
            result["sugar_central"][valid],
            color="black",
            linewidth=1.7,
            marker=marker,
            markersize=5.5,
            markerfacecolor="black",
            markeredgecolor="black",
            label=f"Mediana {dataset_label}",
        )

        # Datos experimentales, mismo marcador que su simulación, pero en azul y sin línea.
        ax.scatter(
            t[valid]/24,
            result["sugar_exp"][valid],
            s=48,
            marker=marker,
            color="tab:blue",
            # edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label=f"Datos {dataset_label}",
        )

    ax.set_title(
        f"Validación de azúcares con banda ±{k_std} desviación estándar\n"
        f"Curva central con mediana de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel("Tiempo real desde inicio de fermentación (días)", labelpad=10)
    ax.set_ylabel("Azúcares totales, S = G + F (g/L)", labelpad=10)
    ax.grid(True, alpha=0.30)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper right",
        fontsize=8.5,
        frameon=True,
        ncol=2,
    )

    metrics_lines = []
    for dataset, result in zip(datasets, results_by_dataset):
        metrics_lines.append(
            f"Set {dataset['id']} - {clean_dataset_name(dataset['name'])}: "
            f"RMSE = {result['rmse']:.3f} g/L; "
            f"NRMSE = {100 * result['nrmse']:.2f}%"
        )

    metrics_text = "\n".join(metrics_lines)

    fig.text(
        0.07,
        0.035,
        metrics_text,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92),
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.22,
        top=0.86,
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("VALIDACIÓN DE AZÚCARES CON BANDAS ± DESVIACIÓN ESTÁNDAR")
    print("=" * 80)

    validate_free_param_sample_lengths()

    print("\nConfiguración:")
    print(f"  Datasets validación: {VALIDATION_DATASET_IDS}")
    print(f"  Muestras por parámetro libre: {N_PARAM_SAMPLES}")
    print("  Figuras: ±1 DE y ±2 DE")

    print("\nParámetros libres detectados:")
    print(f"  {FREE_PARAM_NAMES}")

    print("\nMedianas y desviaciones estándar:")
    for name in FREE_PARAM_NAMES:
        lb, ub = BOUNDS_DICT[name]
        print(
            f"  {name}: mediana = {FREE_PARAM_MEDIAN[name]:.8f}, "
            f"std = {FREE_PARAM_STD[name]:.8f}, bounds = [{lb}, {ub}]"
        )

    selected_info = choose_datasets_by_ids(DATASETS_INFO, VALIDATION_DATASET_IDS)

    print("\nCargando datasets:")
    datasets = []
    for item in selected_info:
        print(f"  Dataset {item['id']:02d}: {item['name']}")
        datasets.append(build_dataset(item))

    for k_std in [1, 2]:
        print(f"\nCalculando curvas para banda ±{k_std} desviación estándar:")

        results = []
        for dataset in datasets:
            print(f"  Set {dataset['id']:02d} - {dataset['name']}")
            result = simulate_dataset_with_std_band(dataset, k_std=k_std)
            results.append(result)

            print(
                f"    RMSE = {result['rmse']:.4f} g/L; "
                f"NRMSE = {100 * result['nrmse']:.2f}%"
            )

        plot_std_band_figure(datasets, results, k_std=k_std)

if __name__ == "__main__":
    main()

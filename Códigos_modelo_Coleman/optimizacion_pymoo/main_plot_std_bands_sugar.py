"""
main_plot_std_bands_sugar.py

Validación de azúcares con bandas deterministas usando desviación estándar.
Adaptado al modelo Coleman.

Este script reemplaza las simulaciones Monte Carlo por simulaciones puntuales:
- curva central: mediana de los parámetros libres
- banda k=1: mediana ± 1 desviación estándar en todos los parámetros libres
- banda k=2: mediana ± 2 desviaciones estándar en todos los parámetros libres

Se generan dos figuras:
1) Bandas con ±1 desviación estándar
2) Bandas con ±2 desviaciones estándar

En cada figura se superponen los 4 datasets de validación definidos en
VALIDATION_DATASET_IDS. Se grafica azúcar S y etanol final/correspondiente.

Convención de errores:
- Azúcar: RMSE en g/L y NRMSE en %.
- Etanol: error absoluto en la misma unidad de Et_final_exp, definido aquí como %. El error relativo se reporta como %.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion_coleman import data_for_simulation, simulate_system
from pymoo_opt_coleman import PARAM_ORDER


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

VALIDATION_DATASET_IDS = [3, 4, 11, 14]
TITLE_WRAP_WIDTH = 42

# Marcadores para distinguir datasets superpuestos
DATASET_MARKERS = ["o", "s", "^", "*"]

# Opciones:
# "single"   -> un solo gráfico con todos los datasets superpuestos
# "subplots" -> figura 2x2, un gráfico por dataset
PLOT_MODE = "subplots"

# Estadístico central a usar para la curva central: "median" o "mean"
CENTRAL_STAT = "median"

# Texto mostrado en figuras/etiquetas según estadístico central
CENTRAL_STAT_DISPLAY = "mediana" if CENTRAL_STAT == "median" else "media"

# Unidades que se muestran en gráficos e impresión.
# Cambia ETHANOL_UNIT a "g/L" si tus datos de etanol final están expresados en g/L.
SUGAR_UNIT = "g/L"
ETHANOL_UNIT = "g/L"


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

# PARÁMETROS FIJOS

FIXED_PARAMS = {
    "mu0": 1.0,
    "kd0": 1.0,
    "betaS0": 1.0,
}


# ============================================================
# UTILIDADES DE PARÁMETROS
# ============================================================

def compute_free_param_statistics(free_param_samples):
    """Calcula media, mediana y desviación estándar muestral para cada parámetro libre.

    Retorna tres diccionarios: (means, medians, stds).
    """

    free_param_mean = {}
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

        free_param_mean[name] = float(np.nanmean(arr))
        free_param_median[name] = float(np.nanmedian(arr))
        free_param_std[name] = float(np.nanstd(arr, ddof=1))

    return free_param_mean, free_param_median, free_param_std


FREE_PARAM_NAMES = list(FREE_PARAM_SAMPLES.keys())
FREE_PARAM_MEAN, FREE_PARAM_MEDIAN, FREE_PARAM_STD = compute_free_param_statistics(FREE_PARAM_SAMPLES)
if CENTRAL_STAT not in ("median", "mean"):
    raise ValueError("CENTRAL_STAT debe ser 'median' o 'mean'.")

# Diccionario con el estadístico central elegido (media o mediana)
FREE_PARAM_CENTER = FREE_PARAM_MEDIAN if CENTRAL_STAT == "median" else FREE_PARAM_MEAN
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


def build_central_param_dict():
    return build_param_dict_from_free_values(FREE_PARAM_CENTER)


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
        median_value = FREE_PARAM_CENTER[name]
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

def compute_validation_cost(sol, sugars_profile, Et_final_exp, penalty=1e12, eps=1e-8):
    """Calcula los costos de validación separados.

    Costo azúcar:
        mean(((S_sim - S_exp) / max(abs(S_exp)))**2)

    Costo etanol:
        ((E_final_sim - E_final_exp) / abs(E_final_exp))**2

    Costo total:
        costo_azúcar + costo_etanol

    Nota: estos costos son adimensionales porque están normalizados.
    No son errores en g/L ni en %.

    Retorna una tupla (costo_azúcar, costo_etanol, costo_total).
    """
    y = sol.y.T
    sugars_sim = np.asarray(y[:, 2], dtype=float)
    Et_final_sim = float(y[-1, 3])

    sugars_profile = np.asarray(sugars_profile, dtype=float)
    Et_final_exp = float(Et_final_exp)

    if len(sugars_sim) != len(sugars_profile):
        return penalty, penalty, penalty

    if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
        return penalty, penalty, penalty

    sugar_scale = max(np.max(np.abs(sugars_profile)), eps)
    ethanol_scale = max(abs(Et_final_exp), eps)

    sugar_res = (sugars_sim - sugars_profile) / sugar_scale
    etoh_res = (Et_final_sim - Et_final_exp) / ethanol_scale

    sugar_error_mean = float(np.mean(sugar_res ** 2))
    ethanol_error = float(etoh_res ** 2)
    objective_total = float(sugar_error_mean + ethanol_error)

    return sugar_error_mean, ethanol_error, objective_total


def compute_validation_costs_for_datasets(datasets, median_params_dict):
    """Calcula los costos de validación separados (azúcar, etanol, total) para todos los datasets.
    
    Retorna tres diccionarios: costs_sugar, costs_ethanol, costs_total.
    """
    costs_sugar = {}
    costs_ethanol = {}
    costs_total = {}
    
    for dataset in datasets:
        try:
            # Preparar parámetros
            params_vector = build_param_vector(median_params_dict)
            x0_og = np.asarray(dataset["x0"], dtype=float)
            
            # Extraer estados desde x0
            X0 = x0_og[0]
            N0 = x0_og[1]
            E0 = x0_og[3]
            
            # Preparar estado inicial Coleman: [X, N, S, E]
            if dataset["sugar_initial"] is not None:
                S0 = float(dataset["sugar_initial"])
            else:
                S0 = 0.0
            
            x0 = np.array([X0, N0, S0, E0], dtype=float)
            
            # Simular con los parámetros medianos
            sol = simulate_system(
                x0=x0,
                t_rel=dataset["t_rel"],
                temp=dataset["temp"],
                Nadd=dataset["Nadd"],
                tspan=dataset["t_span"],
                params_list=params_vector,
            )
            
            # Calcular los costos (retorna tupla: sugar, ethanol, total)
            sugar_cost, ethanol_cost, total_cost = compute_validation_cost(
                sol,
                dataset["sugars_profile"],
                dataset["Et_final_exp"],
            )
            
            costs_sugar[dataset["id"]] = float(sugar_cost)
            costs_ethanol[dataset["id"]] = float(ethanol_cost)
            costs_total[dataset["id"]] = float(total_cost)
            
        except Exception:
            costs_sugar[dataset["id"]] = np.nan
            costs_ethanol[dataset["id"]] = np.nan
            costs_total[dataset["id"]] = np.nan
    
    return costs_sugar, costs_ethanol, costs_total


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
    """Simula un dataset y retorna tiempo y azúcar total S = G + F."""

    params_vector = build_param_vector(params_dict)
    x0_og = np.asarray(dataset["x0"], dtype=float)

    # Extraer estados desde el x0 de 4 estados: [X, N, S, E]
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
        params_list=params_vector,
    )

    y = sol.y.T
    sugars = np.asarray(y[:, 2], dtype=float)

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

    central_params = build_central_param_dict()
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


def simulate_dataset_ethanol_with_std_band(dataset, k_std):
    """
    Para un dataset y un k_std, calcula:
    - curva etanol central con mediana
    - curva etanol con mediana + k_std*std
    - curva etanol con mediana - k_std*std
    - banda low/high entre ambas curvas extremas

    Todo interpolado a los tiempos experimentales.
    """
    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    
    central_params = build_central_param_dict()
    plus_params = build_std_shift_param_dict(k_std=k_std, sign=1)
    minus_params = build_std_shift_param_dict(k_std=k_std, sign=-1)

    # Simular cada escenario
    params_vector_central = build_param_vector(central_params)
    params_vector_plus = build_param_vector(plus_params)
    params_vector_minus = build_param_vector(minus_params)
    
    x0_og = np.asarray(dataset["x0"], dtype=float)
    X0, N0, E0 = x0_og[0], x0_og[1], x0_og[3]
    S0 = float(dataset["sugar_initial"]) if dataset["sugar_initial"] is not None else 0.0
    x0 = np.array([X0, N0, S0, E0], dtype=float)
    
    sol_central = simulate_system(
        x0=x0, t_rel=dataset["t_rel"], temp=dataset["temp"], Nadd=dataset["Nadd"],
        tspan=dataset["t_span"], params_list=params_vector_central
    )
    sol_plus = simulate_system(
        x0=x0, t_rel=dataset["t_rel"], temp=dataset["temp"], Nadd=dataset["Nadd"],
        tspan=dataset["t_span"], params_list=params_vector_plus
    )
    sol_minus = simulate_system(
        x0=x0, t_rel=dataset["t_rel"], temp=dataset["temp"], Nadd=dataset["Nadd"],
        tspan=dataset["t_span"], params_list=params_vector_minus
    )
    
    # Extraer curvas de etanol
    t_sim = np.asarray(sol_central.t, dtype=float)
    Et_central_curve = np.asarray(sol_central.y[3, :], dtype=float)
    Et_plus_curve = np.asarray(sol_plus.y[3, :], dtype=float)
    Et_minus_curve = np.asarray(sol_minus.y[3, :], dtype=float)
    
    # Interpolar a tiempos experimentales
    Et_central = np.interp(t_exp, t_sim, Et_central_curve)
    Et_plus = np.interp(t_exp, t_sim, Et_plus_curve)
    Et_minus = np.interp(t_exp, t_sim, Et_minus_curve)
    
    # Banda de incertidumbre
    band_low = np.minimum(Et_plus, Et_minus)
    band_high = np.maximum(Et_plus, Et_minus)
    
    # Valores finales
    Et_central_final = float(Et_central[-1])
    Et_exp = float(dataset["Et_final_exp"])
    
    # Calcular errores basados en valores finales.
    # Diferencia firmada: indica si la simulación sobreestima (+) o subestima (-).
    # Error absoluto: magnitud de la diferencia en la misma unidad de Et_exp.
    # Error relativo absoluto: error_abs / abs(Et_exp), reportado luego como porcentaje.
    error_signed = Et_central_final - Et_exp
    error_abs = abs(error_signed)
    error_rel = error_abs / abs(Et_exp) if abs(Et_exp) > 1e-8 else np.nan
    
    # Marcar valores válidos
    valid = np.isfinite(t_exp) & np.isfinite(Et_central) & np.isfinite(Et_exp)
    
    return {
        "t_exp": t_exp,
        "Et_central": Et_central,
        "Et_plus": Et_plus,
        "Et_minus": Et_minus,
        "band_low": band_low,
        "band_high": band_high,
        "Et_exp": Et_exp,
        "Et_central_final": Et_central_final,
        "error_signed": error_signed,
        "error_abs": error_abs,
        "error_rel": error_rel,
        "valid": valid,
    }


# ============================================================
# GRÁFICOS
# ============================================================

# Estilos de bandas:
# - ±2 DE se grafica primero, más suave, para mostrar el rango amplio.
# - ±1 DE se grafica encima, más marcada, para resaltar la zona más cercana a la mediana.
BAND_STYLE_2STD = {
    "color": "#f08080",
    "alpha": 0.14,
    "linewidth": 0,
}

BAND_STYLE_1STD = {
    "color": "#d62728",
    "alpha": 0.32,
    "linewidth": 0,
}


def _deduplicate_legend(ax, **legend_kwargs):
    """Evita etiquetas repetidas en la leyenda."""
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), **legend_kwargs)


def plot_dataset_with_two_bands_on_axis(
    ax,
    dataset,
    result_1std,
    result_2std,
    idx=0,
    show_dataset_label=True,
    show_band_labels=True,
    val_cost=None,
):
    """Grafica azúcares con bandas ±1 DE y ±2 DE en el mismo eje."""
    marker = DATASET_MARKERS[idx % len(DATASET_MARKERS)]

    t = result_1std["t_exp"]
    valid = result_1std["valid"]
    dataset_label = f"Set {dataset['id']}"

    # Banda amplia: ±2 DE, más suave y al fondo.
    ax.fill_between(
        t[valid] / 24,
        result_2std["band_low"][valid],
        result_2std["band_high"][valid],
        label="Banda ±2 DE" if show_band_labels else "_nolegend_",
        zorder=1,
        **BAND_STYLE_2STD,
    )

    # Banda interna: ±1 DE, más marcada y encima.
    ax.fill_between(
        t[valid] / 24,
        result_1std["band_low"][valid],
        result_1std["band_high"][valid],
        label="Banda ±1 DE" if show_band_labels else "_nolegend_",
        zorder=2,
        **BAND_STYLE_1STD,
    )

    ax.plot(
        t[valid] / 24,
        result_1std["sugar_central"][valid],
        color="black",
        linewidth=1.7,
        marker=marker,
        markersize=5.5,
        markerfacecolor="black",
        markeredgecolor="black",
        label=f"{CENTRAL_STAT_DISPLAY.capitalize()} parámetros" if not show_dataset_label else f"{CENTRAL_STAT_DISPLAY.capitalize()} {dataset_label}",
        zorder=3,
    )

    ax.scatter(
        t[valid] / 24,
        result_1std["sugar_exp"][valid],
        s=48,
        marker=marker,
        color="tab:blue",
        linewidth=0.6,
        zorder=4,
        label="Datos experimentales" if not show_dataset_label else f"Datos {dataset_label}",
    )

    ax.set_xlabel("Tiempo real desde inicio de fermentación (días)")
    ax.set_ylabel(f"Azúcares totales, S = G + F ({SUGAR_UNIT})")
    ax.grid(True, alpha=0.30)


def plot_ethanol_with_two_bands_on_axis(
    ax,
    dataset,
    result_1std,
    result_2std,
    idx=0,
    show_dataset_label=True,
    show_band_labels=True,
    val_cost=None,
):
    """Grafica etanol con bandas ±1 DE y ±2 DE en el mismo eje."""
    marker = DATASET_MARKERS[idx % len(DATASET_MARKERS)]
    dataset_label = f"Set {dataset['id']}"

    t = result_1std["t_exp"] / 24
    valid = result_1std["valid"]

    # Banda amplia: ±2 DE, más suave y al fondo.
    ax.fill_between(
        t[valid],
        result_2std["band_low"][valid],
        result_2std["band_high"][valid],
        label="Banda ±2 DE" if show_band_labels else "_nolegend_",
        zorder=1,
        **BAND_STYLE_2STD,
    )

    # Banda interna: ±1 DE, más marcada y encima.
    ax.fill_between(
        t[valid],
        result_1std["band_low"][valid],
        result_1std["band_high"][valid],
        label="Banda ±1 DE" if show_band_labels else "_nolegend_",
        zorder=2,
        **BAND_STYLE_1STD,
    )

    ax.plot(
        t[valid],
        result_1std["Et_central"][valid],
        color="black",
        linewidth=1.7,
        marker=marker,
        markersize=5.5,
        markerfacecolor="black",
        markeredgecolor="black",
        label=f"{CENTRAL_STAT_DISPLAY.capitalize()} parámetros" if not show_dataset_label else f"{CENTRAL_STAT_DISPLAY.capitalize()} {dataset_label}",
        zorder=3,
    )

    ax.plot(
        t[valid][-1],
        result_1std["Et_exp"],
        color="tab:blue",
        linewidth=2,
        marker=marker,
        markersize=8,
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        label="Dato experimental final" if not show_dataset_label else f"Dato final {dataset_label}",
        zorder=5,
    )

    ax.set_xlabel("Tiempo real desde inicio de fermentación (días)")
    ax.set_ylabel(f"Etanol final ({ETHANOL_UNIT})")
    ax.grid(True, alpha=0.30)


def plot_std_bands_combined_figure(
    datasets,
    results_1std_by_dataset,
    results_2std_by_dataset,
    plot_mode="single",
    validation_costs=None,
):
    """Genera una única figura de azúcares con bandas ±1 DE y ±2 DE."""
    if validation_costs is None:
        validation_costs = {}

    if plot_mode == "single":
        fig, ax = plt.subplots(figsize=(14, 9))

        for idx, (dataset, result_1std, result_2std) in enumerate(
            zip(datasets, results_1std_by_dataset, results_2std_by_dataset)
        ):
            plot_dataset_with_two_bands_on_axis(
                ax=ax,
                dataset=dataset,
                result_1std=result_1std,
                result_2std=result_2std,
                idx=idx,
                show_dataset_label=True,
                show_band_labels=(idx == 0),
                val_cost=validation_costs.get(dataset["id"], None),
            )

        ax.set_title(
            "Validación de azúcares con bandas ±1 y ±2 desviaciones estándar\n"
            f"Curva central con {CENTRAL_STAT_DISPLAY} de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
            fontsize=14,
            pad=14,
        )

        _deduplicate_legend(
            ax,
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            ncol=2,
        )

        metrics_lines = []
        for dataset, result in zip(datasets, results_1std_by_dataset):
            line = (
                f"Set {dataset['id']} - {clean_dataset_name(dataset['name'])}: "
                f"RMSE = {result['rmse']:.3f} {SUGAR_UNIT}; "
                f"NRMSE = {100 * result['nrmse']:.2f}%"
            )
            if dataset["id"] in validation_costs:
                line += f"; Costo azúcar = {validation_costs[dataset['id']]: .4f}"
            metrics_lines.append(line)

        fig.text(
            0.07,
            0.035,
            "\n".join(metrics_lines),
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

    elif plot_mode == "subplots":
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, (ax, dataset, result_1std, result_2std) in enumerate(
            zip(axes, datasets, results_1std_by_dataset, results_2std_by_dataset)
        ):
            plot_dataset_with_two_bands_on_axis(
                ax=ax,
                dataset=dataset,
                result_1std=result_1std,
                result_2std=result_2std,
                idx=idx,
                show_dataset_label=False,
                show_band_labels=True,
                val_cost=validation_costs.get(dataset["id"], None),
            )

            title = (
                f"Set {dataset['id']} - {clean_dataset_name(dataset['name'])}\n"
                f"RMSE = {result_1std['rmse']:.3f} {SUGAR_UNIT} | "
                f"NRMSE = {100 * result_1std['nrmse']:.2f}%"
            )
            if dataset["id"] in validation_costs:
                title += f" | Costo azúcar = {validation_costs[dataset['id']]: .4f}"

            ax.set_title(title, fontsize=10.5, pad=10)
            _deduplicate_legend(ax, loc="best", fontsize=8, frameon=True)

        for ax in axes[len(datasets):]:
            ax.axis("off")

        fig.suptitle(
            "Validación de azúcares con bandas ±1 y ±2 desviaciones estándar\n"
            f"Curva central con {CENTRAL_STAT_DISPLAY} de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
            fontsize=14,
            y=0.98,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    else:
        raise ValueError("plot_mode debe ser 'single' o 'subplots'.")


def plot_ethanol_combined_figure(
    datasets,
    results_1std_by_dataset,
    results_2std_by_dataset,
    plot_mode="single",
    validation_costs=None,
):
    """Genera una única figura de etanol con bandas ±1 DE y ±2 DE."""
    if validation_costs is None:
        validation_costs = {}

    if plot_mode == "single":
        fig, ax = plt.subplots(figsize=(14, 9))

        for idx, (dataset, result_1std, result_2std) in enumerate(
            zip(datasets, results_1std_by_dataset, results_2std_by_dataset)
        ):
            plot_ethanol_with_two_bands_on_axis(
                ax=ax,
                dataset=dataset,
                result_1std=result_1std,
                result_2std=result_2std,
                idx=idx,
                show_dataset_label=True,
                show_band_labels=(idx == 0),
                val_cost=validation_costs.get(dataset["id"], None),
            )

        ax.set_title(
            "Validación de etanol final con bandas ±1 y ±2 desviaciones estándar\n"
            f"Valor central con {CENTRAL_STAT_DISPLAY} de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
            fontsize=14,
            pad=14,
        )

        _deduplicate_legend(
            ax,
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            ncol=2,
        )

        metrics_lines = []
        for dataset, result in zip(datasets, results_1std_by_dataset):
            line = (
                f"Set {dataset['id']} - {clean_dataset_name(dataset['name'])}: "
                f"Error abs = {result['error_abs']:.3f} {ETHANOL_UNIT}; "
                f"Error rel = {100 * result['error_rel']:.2f}%"
            )
            if dataset["id"] in validation_costs:
                line += f"; Costo etanol = {validation_costs[dataset['id']]: .4f}"
            metrics_lines.append(line)

        fig.text(
            0.07,
            0.035,
            "\n".join(metrics_lines),
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

    elif plot_mode == "subplots":
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, (ax, dataset, result_1std, result_2std) in enumerate(
            zip(axes, datasets, results_1std_by_dataset, results_2std_by_dataset)
        ):
            plot_ethanol_with_two_bands_on_axis(
                ax=ax,
                dataset=dataset,
                result_1std=result_1std,
                result_2std=result_2std,
                idx=idx,
                show_dataset_label=False,
                show_band_labels=True,
                val_cost=validation_costs.get(dataset["id"], None),
            )

            title = (
                f"Set {dataset['id']} - {clean_dataset_name(dataset['name'])}\n"
                f"Error abs = {result_1std['error_abs']:.3f} {ETHANOL_UNIT} | "
                f"Error rel = {100 * result_1std['error_rel']:.2f}%"
            )
            if dataset["id"] in validation_costs:
                title += f" | Costo etanol = {validation_costs[dataset['id']]: .4f}"

            ax.set_title(title, fontsize=10.5, pad=10)
            _deduplicate_legend(ax, loc="best", fontsize=8, frameon=True)

        for ax in axes[len(datasets):]:
            ax.axis("off")

        fig.suptitle(
            "Validación de etanol final con bandas ±1 y ±2 desviaciones estándar\n"
                f"Valor central con {CENTRAL_STAT_DISPLAY} de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
            fontsize=14,
            y=0.98,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    else:
        raise ValueError("plot_mode debe ser 'single' o 'subplots'.")


# ============================================================
# MAIN
# ============================================================

def main():
    validate_free_param_sample_lengths()

    print("Validación de azúcares y etanol con bandas ±1 y ±2 desviaciones estándar")

    selected_info = choose_datasets_by_ids(DATASETS_INFO, VALIDATION_DATASET_IDS)
    datasets = []
    for item in selected_info:
        datasets.append(build_dataset(item))

    # Calcular costos de validación separados.
    median_params_dict = build_central_param_dict()
    costs_sugar, costs_ethanol, costs_total = compute_validation_costs_for_datasets(datasets, median_params_dict)

    sugar_results_by_k = {1: [], 2: []}
    ethanol_results_by_k = {1: [], 2: []}

    for k_std in [1, 2]:
        for dataset in datasets:
            result = simulate_dataset_with_std_band(dataset, k_std=k_std)
            sugar_results_by_k[k_std].append(result)

        for dataset in datasets:
            result = simulate_dataset_ethanol_with_std_band(dataset, k_std=k_std)
            ethanol_results_by_k[k_std].append(result)

    plot_std_bands_combined_figure(
        datasets=datasets,
        results_1std_by_dataset=sugar_results_by_k[1],
        results_2std_by_dataset=sugar_results_by_k[2],
        plot_mode=PLOT_MODE,
        validation_costs=costs_sugar,
    )

    plot_ethanol_combined_figure(
        datasets=datasets,
        results_1std_by_dataset=ethanol_results_by_k[1],
        results_2std_by_dataset=ethanol_results_by_k[2],
        plot_mode=PLOT_MODE,
        validation_costs=costs_ethanol,
    )

if __name__ == "__main__":
    main()


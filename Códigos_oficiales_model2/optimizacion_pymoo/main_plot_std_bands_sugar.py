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

# Modo de visualización:
# "single"   -> un solo gráfico con todos los datasets superpuestos
# "subplots" -> una figura con varios subgráficos, uno por dataset
PLOT_MODE = "subplots"


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
    "mu0": [90.34555249, 89.52184298, 95.23921271, 99.86220292, 96.21559779],
    "betaS0": [26.51646471, 15.99448248, 25.10324653, 22.67502847, 25.22095397],
    "Yes": [0.354987563, 0.343230535, 0.36512106, 0.360495043, 0.372827078],
    "Ks": [72.01803233, 72.36568508, 99.97551222, 62.83038446, 65.84344506],
}


#

FIXED_PARAMS = {
    "kd0": 0.001014347,
    "Kn": 94.83746055,
    "Yxn": 90.11679681,
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

    # Extraer estados desde el x0 antiguo
    X0 = x0_og[0]
    N0 = x0_og[1]
    E0 = x0_og[3]

    # Cambiar azúcares
    if dataset["sugar_initial"] is not None:
        S0 = float(dataset["sugar_initial"])
    else:
        S0 = 0.0

    sugar0 = S0

    x0 = np.array([X0, N0, sugar0, E0], dtype=float)

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

def plot_dataset_on_axis(ax, dataset, result, k_std, idx=0, show_dataset_label=True):
    marker = DATASET_MARKERS[idx % len(DATASET_MARKERS)]
    t = result["t_exp"]
    valid = result["valid"]

    dataset_label = f"Set {dataset['id']}"

    ax.fill_between(
        t[valid] / 24,
        result["band_low"][valid],
        result["band_high"][valid],
        color="#f08080",
        alpha=0.18,
        linewidth=0,
        label=f"Banda ±{k_std} DE" if idx == 0 else None,
    )

    ax.plot(
        t[valid] / 24,
        result["sugar_central"][valid],
        color="black",
        linewidth=1.7,
        marker=marker,
        markersize=5.5,
        markerfacecolor="black",
        markeredgecolor="black",
        label="Mediana parámetros" if not show_dataset_label else f"Mediana {dataset_label}",
    )

    ax.scatter(
        t[valid] / 24,
        result["sugar_exp"][valid],
        s=48,
        marker=marker,
        color="tab:blue",
        linewidth=0.6,
        zorder=4,
        label="Datos experimentales" if not show_dataset_label else f"Datos {dataset_label}",
    )

    ax.set_xlabel("Tiempo real desde inicio de fermentación (días)")
    ax.set_ylabel("Azúcares totales, S = G + F (g/L)")
    ax.grid(True, alpha=0.30)


def plot_std_band_figure(datasets, results_by_dataset, k_std, plot_mode="single"):
    if plot_mode == "single":
        fig, ax = plt.subplots(figsize=(14, 8.5))

        for idx, (dataset, result) in enumerate(zip(datasets, results_by_dataset)):
            plot_dataset_on_axis(
                ax=ax,
                dataset=dataset,
                result=result,
                k_std=k_std,
                idx=idx,
                show_dataset_label=True,
            )

        ax.set_title(
            f"Validación de azúcares con banda ±{k_std} desviación estándar\n"
            f"Curva central con mediana de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
            fontsize=14,
            pad=14,
        )

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
        fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, (ax, dataset, result) in enumerate(zip(axes, datasets, results_by_dataset)):
            plot_dataset_on_axis(
                ax=ax,
                dataset=dataset,
                result=result,
                k_std=k_std,
                idx=idx,
                show_dataset_label=False,
            )
            ax.set_title(
                f"Set {dataset['id']} - {clean_dataset_name(dataset['name'])}\n"
                f"RMSE = {result['rmse']:.3f} g/L | NRMSE = {100 * result['nrmse']:.2f}%",
                fontsize=10.5,
                pad=10,
            )
            ax.legend(loc="best", fontsize=8, frameon=True)

        for ax in axes[len(datasets):]:
            ax.axis("off")

        fig.suptitle(
            f"Validación de azúcares con banda ±{k_std} desviación estándar\n"
            f"Curva central con mediana de parámetros; n = {N_PARAM_SAMPLES} muestras por parámetro",
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
    print("=" * 80)
    print("VALIDACIÓN DE AZÚCARES CON BANDAS ± DESVIACIÓN ESTÁNDAR")
    print("=" * 80)

    validate_free_param_sample_lengths()

    print("\nConfiguración:")
    print(f"  Datasets validación: {VALIDATION_DATASET_IDS}")
    print(f"  Muestras por parámetro libre: {N_PARAM_SAMPLES}")
    print("  Figuras: ±1 DE y ±2 DE")
    print(f"  Modo de gráficos: {PLOT_MODE}")

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

        plot_std_band_figure(datasets, results, k_std=k_std, plot_mode=PLOT_MODE)

if __name__ == "__main__":
    main()

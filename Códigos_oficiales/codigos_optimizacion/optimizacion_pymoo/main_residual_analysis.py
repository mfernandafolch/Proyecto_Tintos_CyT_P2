import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

# Suppress FutureWarning from scipy.stats.anderson about method parameter (SciPy >= 1.17)
warnings.filterwarnings("ignore", message=".*anderson.*method.*")

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main_plot_std_bands_sugar import (
    FREE_PARAM_NAMES,
    DATASETS_INFO,
    VALIDATION_DATASET_IDS,
    build_dataset,
    build_median_param_dict,
    build_param_vector,
    choose_datasets_by_ids,
)

from simulacion import simulate_system


# ============================================================
# DATASETS DE VALIDACIÓN
# ============================================================

validation_datasets = [
    build_dataset(item)
    for item in choose_datasets_by_ids(DATASETS_INFO, VALIDATION_DATASET_IDS)
]

best_params_dict = build_median_param_dict()


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUN_RESIDUAL_ANALYSIS = True
PLOT_RESIDUALS = True

# p = número de parámetros del modelo estimados/calibrados
P = len(FREE_PARAM_NAMES)


# ============================================================
# CONSTRUCCIÓN DE X0 PERSONALIZADO
# ============================================================

def build_custom_x0(dataset):
    """
    Construye x0 usando:
    - X0, N0 y E0 desde dataset["x0"]
    - G0 y F0 desde dataset["sugar_initial"]

    Modelo:
    x = [X, N, G, F, E]
    """

    x0_og = np.asarray(dataset["x0"], dtype=float)

    X0 = x0_og[0]
    N0 = x0_og[1]
    E0 = x0_og[4]

    if dataset.get("sugar_initial", None) is not None:
        S0 = float(dataset["sugar_initial"])
    else:
        S0 = float(x0_og[2] + x0_og[3])

    G0 = S0 / 2
    F0 = S0 / 2

    return np.array([X0, N0, G0, F0, E0], dtype=float)


# ============================================================
# SIMULACIÓN
# ============================================================

def simulate_dataset_for_residuals(dataset, params_dict):
    """
    Simula un dataset usando x0 personalizado.
    Retorna:
    - t_sim
    - S_sim = G + F
    """

    params_vector = build_param_vector(params_dict)
    x0 = build_custom_x0(dataset)

    sol = simulate_system(
        x0=x0,
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=params_vector,
    )

    t_sim = sol.t
    G_sim = sol.y[2, :]
    F_sim = sol.y[3, :]
    S_sim = G_sim + F_sim

    return t_sim, S_sim


# ============================================================
# MÉTRICAS DE ERROR SEGÚN DIAPOSITIVAS
# ============================================================

def compute_error_quantification(y_exp, y_model, p=0):
    """
    Métricas de error para análisis de residuos.

    Notación:
    e_i = y_i - y_hat_i

    Donde:
    y_i     = valor experimental
    y_hat_i = valor del modelo

    Métricas:
    - MSE
    - RMSE
    - NRMSE
    - R2
    - adjusted R2
    """

    y_exp = np.asarray(y_exp, dtype=float)
    y_model = np.asarray(y_model, dtype=float)

    valid = np.isfinite(y_exp) & np.isfinite(y_model)

    y_exp = y_exp[valid]
    y_model = y_model[valid]

    residuals = y_exp - y_model

    n = len(residuals)

    SSE = np.sum(residuals**2)
    SST = np.sum((y_exp - np.mean(y_exp))**2)

    # MSE de la diapositiva:
    # MSE = sum(e_i^2) / (n - p)
    MSE = SSE / (n - p) if n > p else np.nan

    RMSE = np.sqrt(np.mean(residuals**2))

    y_range = np.max(y_exp) - np.min(y_exp)
    NRMSE = RMSE / y_range if y_range > 0 else np.nan

    # Determination coefficient - R2
    R2 = 1 - SSE / SST if SST > 0 else np.nan

    # adjusted R2
    # R2_adj = 1 - [(1 - R2)(n - 1)] / (n - p - 1)
    adjusted_R2 = (
        1 - ((1 - R2) * (n - 1)) / (n - p - 1)
        if n > p + 1 and np.isfinite(R2)
        else np.nan
    )

    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals, ddof=1) if n > 1 else np.nan

    return {
        "n": n,
        "p": p,
        "residuals": residuals,

        # Nombres de las diapositivas
        "MSE": MSE,
        "R2": R2,
        "adjusted_R2": adjusted_R2,

        # Métricas complementarias útiles
        "RMSE": RMSE,
        "NRMSE": NRMSE,
        "SSE": SSE,
        "SST": SST,
        "mean_residual": mean_residual,
        "std_residual": std_residual,
    }


# ============================================================
# NORMALITY: ANDERSON-DARLING TEST
# ============================================================

def anderson_darling_test(residuals, alpha=5):
    """
    Anderson-Darling test.

    H0: The data follow the specified distribution.
    H1: The data do not follow the specified distribution.

    Si A2 statistic > critical value:
        Se rechaza H0.
    """

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 3:
        return {
            "A2_statistic": np.nan,
            "critical_value": np.nan,
            "alpha": alpha,
            "H0": "The data follow the specified distribution",
            "H1": "The data do not follow the specified distribution",
            "decision": "No hay suficientes datos",
            "normality_result": "No evaluable",
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*anderson.*method.*")
        result = stats.anderson(residuals, dist="norm")

    idx = np.argmin(np.abs(result.significance_level - alpha))
    critical_value = result.critical_values[idx]

    reject_H0 = result.statistic > critical_value

    return {
        "A2_statistic": result.statistic,
        "critical_value": critical_value,
        "alpha": alpha,
        "H0": "The data follow the specified distribution",
        "H1": "The data do not follow the specified distribution",
        "decision": "Reject H0" if reject_H0 else "Do not reject H0",
        "normality_result": (
            "Residuals do not distribute normally"
            if reject_H0
            else "Residuals distribute normally"
        ),
    }


# ============================================================
# AUTOCORRELATION: DURBIN-WATSON TEST
# ============================================================

def durbin_watson_test(residuals):
    """
    Durbin-Watson test.

    d cercano a 2 indica independencia.
    d cercano a 0 indica autocorrelación positiva.
    d cercano a 4 indica autocorrelación negativa.
    """

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 3:
        return {
            "d_statistic": np.nan,
            "autocorrelation_result": "No hay suficientes datos",
        }

    d = durbin_watson(residuals)

    if d < 1.5:
        autocorrelation_result = "Positive autocorrelation"
    elif d > 2.5:
        autocorrelation_result = "Negative autocorrelation"
    else:
        autocorrelation_result = "No strong evidence of autocorrelation"

    return {
        "d_statistic": d,
        "autocorrelation_result": autocorrelation_result,
    }


# ============================================================
# ANÁLISIS DE RESIDUOS PARA UN DATASET
# ============================================================

def analyze_residuals_dataset(dataset, params_dict, p=0):
    """
    Ejecuta análisis de residuos para un dataset de validación.
    """

    t_sim, S_sim = simulate_dataset_for_residuals(dataset, params_dict)

    t_exp = np.asarray(dataset["t_rel"], dtype=float)
    S_exp = np.asarray(dataset["sugars_profile"], dtype=float)

    S_model = np.interp(t_exp, t_sim, S_sim)

    error_quantification = compute_error_quantification(
        y_exp=S_exp,
        y_model=S_model,
        p=p,
    )

    residuals = error_quantification["residuals"]

    normality = anderson_darling_test(residuals)
    autocorrelation = durbin_watson_test(residuals)

    return {
        "dataset_id": dataset["id"],
        "dataset_name": dataset["name"],

        "t_exp": t_exp,
        "y_exp": S_exp,
        "y_model": S_model,

        "residuals": residuals,

        "error_quantification": error_quantification,
        "normality_Anderson_Darling": normality,
        "autocorrelation_Durbin_Watson": autocorrelation,
    }


# ============================================================
# GRÁFICO DE ANÁLISIS DE RESIDUOS
# ============================================================

def plot_residual_analysis(result):
    """
    Figura con herramientas de análisis de residuos:
    1. Residuals vs time
    2. Histogram of residuals
    3. Normal probability plot
    4. Lagged residuals plot
    """

    t_exp = result["t_exp"]
    residuals = result["residuals"]

    error = result["error_quantification"]
    normality = result["normality_Anderson_Darling"]
    autocorrelation = result["autocorrelation_Durbin_Watson"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --------------------------------------------------------
    # Residuals vs time
    # --------------------------------------------------------
    ax = axes[0, 0]

    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(t_exp / 24, residuals, s=45)
    ax.plot(t_exp / 24, residuals, linewidth=1, alpha=0.7)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Residual, eᵢ = yᵢ - ŷᵢ (g/L)")
    ax.set_title("Residuals vs time")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Histogram of residuals
    # --------------------------------------------------------
    ax = axes[0, 1]

    ax.hist(residuals, bins="auto", density=True, alpha=0.65)

    if len(residuals) >= 2:
        mu, std = stats.norm.fit(residuals)
        x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        pdf = stats.norm.pdf(x, mu, std)
        ax.plot(x, pdf, color="black", linewidth=2, label="Normal fit")
        ax.legend(fontsize=8)

    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of residuals")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Normal probability plot
    # --------------------------------------------------------
    ax = axes[1, 0]

    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title("Normal probability plot")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Lagged residuals plot
    # --------------------------------------------------------
    ax = axes[1, 1]

    if len(residuals) >= 2:
        ax.scatter(residuals[:-1], residuals[1:], s=45)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

    ax.set_xlabel("eₜ")
    ax.set_ylabel("eₜ₊₁")
    ax.set_title("Lagged residuals plot")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Analysis of Residuals - Set {result['dataset_id']}\n"
        f"MSE = {error['MSE']:.3f} | "
        f"R² = {error['R2']:.3f} | "
        f"adjusted R² = {error['adjusted_R2']:.3f} | "
        f"Anderson-Darling A² = {normality['A2_statistic']:.3f} "
        f"/ critical value = {normality['critical_value']:.3f} "
        f"({normality['normality_result']}) | "
        f"Durbin-Watson d = {autocorrelation['d_statistic']:.3f} "
        f"({autocorrelation['autocorrelation_result']})",
        fontsize=11.5,
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


# ============================================================
# ANÁLISIS PARA TODOS LOS DATASETS DE VALIDACIÓN
# ============================================================

def run_residual_analysis(validation_datasets, params_dict):
    """
    Corre analysis of residuals para todos los datasets de validación.
    """

    all_results = []

    for dataset in validation_datasets:

        result = analyze_residuals_dataset(
            dataset=dataset,
            params_dict=params_dict,
            p=P,
        )

        all_results.append(result)

        if PLOT_RESIDUALS:
            plot_residual_analysis(result)

    return all_results


# ============================================================
# RESUMEN EN CONSOLA
# ============================================================

def print_residual_summary(residual_results):
    """
    Imprime resumen usando nombres de las diapositivas.
    """

    print("\n" + "="*90)
    print("ANALYSIS OF RESIDUALS - MODEL VALIDATION")
    print("="*90)
    print(f"Free parameters: {FREE_PARAM_NAMES}")
    print(f"p = number of model parameters: {P}")

    for result in residual_results:

        error = result["error_quantification"]
        normality = result["normality_Anderson_Darling"]
        autocorrelation = result["autocorrelation_Durbin_Watson"]

        print("\n" + "-"*90)
        print(f"Set {result['dataset_id']} - {result['dataset_name']}")
        print("-"*90)

        print("Residual:")
        print("  e_i = (experimental value)_i - (model value)_i")
        print(f"  n = number of measurements: {error['n']}")
        print(f"  p = number of model parameters: {error['p']}")

        print("\nError quantification:")
        print(f"  MSE:         {error['MSE']:.4f}")
        print(f"  RMSE:        {error['RMSE']:.4f} g/L")
        print(f"  NRMSE:       {100*error['NRMSE']:.2f}%")
        print(f"  R2:          {error['R2']:.4f}")
        print(f"  adjusted R2: {error['adjusted_R2']:.4f}")
        print(f"  SSE:         {error['SSE']:.4f}")
        print(f"  Mean residual: {error['mean_residual']:.4f} g/L")
        print(f"  Std residual:  {error['std_residual']:.4f} g/L")

        print("\nNormality - Anderson-Darling test:")
        print(f"  H0: {normality['H0']}")
        print(f"  H1: {normality['H1']}")
        print(f"  A2 statistic:   {normality['A2_statistic']:.4f}")
        print(f"  Critical value: {normality['critical_value']:.4f}")
        print(f"  alpha:          {normality['alpha']}%")
        print(f"  Decision:       {normality['decision']}")
        print(f"  Result:         {normality['normality_result']}")

        print("\nAutocorrelation - Durbin-Watson test:")
        print(f"  d statistic: {autocorrelation['d_statistic']:.4f}")
        print(f"  Result:      {autocorrelation['autocorrelation_result']}")


# ============================================================
# EJECUCIÓN
# ============================================================

if RUN_RESIDUAL_ANALYSIS:

    residual_results = run_residual_analysis(
        validation_datasets=validation_datasets,
        params_dict=best_params_dict,
    )

    print_residual_summary(residual_results)
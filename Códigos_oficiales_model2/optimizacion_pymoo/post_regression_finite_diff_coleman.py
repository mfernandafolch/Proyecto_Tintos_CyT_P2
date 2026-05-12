# -*- coding: utf-8 -*-
"""
post_regression_finite_diff_coleman.py

Análisis post-regresión por diferencias finitas para el modelo Coleman.

Permite analizar distintas combinaciones de parámetros libres/fijos sin usar CasADi.

Evalúa:
    - Sensibilidad media relativa
    - Matriz de correlación entre parámetros
    - FIM aproximada
    - Desviación estándar de parámetros
    - t-values
    - Intervalos de confianza aproximados
    - Recomendación: estimar / revisar / fijar

El análisis usa como salidas observables:
    - S(t): azúcares totales simulados
    - E_final: etanol final simulado
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Rutas e imports
# -------------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from simulacion_coleman import data_for_simulation, simulate_system

try:
    from pymoo_opt_coleman import PARAM_ORDER
except Exception:
    PARAM_ORDER = ["mu0", "kd0", "betaS0", "Kn", "Yxn", "Yes", "Ks"]


# -------------------------------------------------------------------------
# CONFIGURACIÓN QUE DEBES EDITAR
# -------------------------------------------------------------------------

paths = [
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",
]


# Vector calibrado base.
# Reemplaza estos valores por tus parámetros estimados con PSO.
params_base_dict = {
  "mu0": 19.790681776545654,
  "kd0": 0.0010143468754638958,
  "betaS0": 40.282534307960944,
  "Kn": 94.83746054756863,
  "Yxn": 90.11679680829091,
  "Yes": 0.36558236678925543,
  "Ks": 36.79611139117356,
}


# Parámetros que quieres analizar como "libres".
# Los que no estén aquí quedan fijos automáticamente en params_base_dict.
free_params_for_analysis = [
    "mu0",
    # "kd0",
    "betaS0",
    # "Kn",
    # "Yxn",
    "Yes",
    "Ks",
]

# Ejemplos:
# free_params_for_analysis = ["mu0", "betaS0", "Yxn", "Yes"]
# free_params_for_analysis = ["mu0", "Yxn", "Yes"]


# Tamaño de perturbación relativa para diferencias finitas.
# 1e-4 suele ser razonable. Si hay ruido numérico, probar 1e-3.
REL_STEP = 1e-4

# Umbrales de decisión
SENSITIVITY_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
TVALUE_THRESHOLD = 2.0

# Si True, guarda resultados en Excel
SAVE_EXCEL = True
OUTPUT_EXCEL = "post_regression_analysis_coleman.xlsx"


# -------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------

def params_dict_to_vector(params_dict, param_order):
    return [float(params_dict[name]) for name in param_order]


def vector_to_params_dict(params_vector, param_order):
    return {name: float(value) for name, value in zip(param_order, params_vector)}


def build_datasets(paths, t_muestreo=3.0):
    datasets = []

    for path in paths:
        x0, t_rel, sugars_profile, temp, Nadd, tspan, Et_final = data_for_simulation(
            excel_path=path,
            t_muestreo=t_muestreo,
        )

        dataset = {
            "path": path,
            "x0": x0,
            "t_rel": t_rel,
            "sugars_profile": sugars_profile,
            "temp": temp,
            "Nadd": Nadd,
            "tspan": tspan,
            "Et_final_exp": Et_final,
        }

        datasets.append(dataset)

        print(f"\nDataset construido: {os.path.basename(path)}")
        print(f"  n datos azúcar: {len(sugars_profile)}")
        print(f"  Etanol final experimental: {Et_final:.4f} g/L")

    return datasets


def simulate_dataset(dataset, params_vector):
    sol = simulate_system(
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["tspan"],
        params_list=params_vector,
    )

    if not sol.success:
        raise RuntimeError(f"Falló la simulación: {sol.message}")

    y = sol.y.T

    S_sim = y[:, 2]
    E_sim = y[:, 3]

    return S_sim, float(E_sim[-1])


def get_observable_vector(dataset, params_vector):
    """
    Vector de salidas observables del modelo.

    Incluye:
        - S(t) completo
        - E_final
    """

    S_sim, E_final_sim = simulate_dataset(dataset, params_vector)

    y_obs = np.concatenate([
        np.asarray(S_sim, dtype=float),
        np.array([E_final_sim], dtype=float),
    ])

    return y_obs


def get_experimental_vector(dataset):
    """
    Vector experimental comparable con get_observable_vector().
    """

    S_exp = np.asarray(dataset["sugars_profile"], dtype=float)
    E_final_exp = float(dataset["Et_final_exp"])

    y_exp = np.concatenate([
        S_exp,
        np.array([E_final_exp], dtype=float),
    ])

    return y_exp


def get_scale_vector(dataset):
    """
    Escala para normalizar sensibilidades relativas.

    Para S(t), usa rango o máximo de azúcares.
    Para E_final, usa valor final experimental.
    """

    S_exp = np.asarray(dataset["sugars_profile"], dtype=float)
    E_final_exp = float(dataset["Et_final_exp"])

    S_scale = max(np.nanmax(S_exp) - np.nanmin(S_exp), np.nanmax(np.abs(S_exp)), 1e-9)
    E_scale = max(abs(E_final_exp), 1e-9)

    scale = np.concatenate([
        np.full(len(S_exp), S_scale, dtype=float),
        np.array([E_scale], dtype=float),
    ])

    return scale


def estimate_measurement_variance(datasets, params_base_vector):
    """
    Estima varianzas desde residuos modelo-dato.

    sigma2_S: varianza de residuos de azúcar
    sigma2_E: varianza de residuos de etanol final
    """

    residuals_S = []
    residuals_E = []

    for dataset in datasets:
        y_sim = get_observable_vector(dataset, params_base_vector)
        y_exp = get_experimental_vector(dataset)

        n_S = len(dataset["sugars_profile"])

        residuals_S.extend(y_exp[:n_S] - y_sim[:n_S])
        residuals_E.append(y_exp[-1] - y_sim[-1])

    residuals_S = np.asarray(residuals_S, dtype=float)
    residuals_E = np.asarray(residuals_E, dtype=float)

    sigma2_S = np.var(residuals_S, ddof=1) if len(residuals_S) > 1 else 1.0
    sigma2_E = np.var(residuals_E, ddof=1) if len(residuals_E) > 1 else sigma2_S

    sigma2_S = max(float(sigma2_S), 1e-12)
    sigma2_E = max(float(sigma2_E), 1e-12)

    return sigma2_S, sigma2_E


def finite_difference_sensitivity(
    datasets,
    params_base_dict,
    param_order,
    free_params,
    rel_step=1e-4,
):
    """
    Calcula matriz de sensibilidad por diferencias finitas centrales.

    G_abs:
        dy/dp

    G_rel:
        sensibilidad relativa general:
        abs(dy/dp * p / scale_y)
    """

    params_base_vector = np.asarray(
        params_dict_to_vector(params_base_dict, param_order),
        dtype=float
    )

    free_indices = [param_order.index(name) for name in free_params]

    G_abs_rows = []
    G_rel_rows = []
    y_exp_all = []
    y_sim_base_all = []
    sigma_labels = []

    print("\n=== Simulación base y sensibilidades ===")

    for d_idx, dataset in enumerate(datasets, start=1):
        print(f"\nDataset {d_idx}: {os.path.basename(dataset['path'])}")

        y_base = get_observable_vector(dataset, params_base_vector)
        y_exp = get_experimental_vector(dataset)
        scale = get_scale_vector(dataset)

        n_outputs = len(y_base)
        n_params = len(free_params)

        G_abs_dataset = np.zeros((n_outputs, n_params), dtype=float)
        G_rel_dataset = np.zeros((n_outputs, n_params), dtype=float)

        for j, param_name in enumerate(free_params):
            p_idx = free_indices[j]
            p_value = params_base_vector[p_idx]

            step = rel_step * max(abs(p_value), 1.0)

            params_plus = params_base_vector.copy()
            params_minus = params_base_vector.copy()

            params_plus[p_idx] = p_value + step
            params_minus[p_idx] = max(p_value - step, 1e-12)

            actual_step = params_plus[p_idx] - params_minus[p_idx]

            y_plus = get_observable_vector(dataset, params_plus)
            y_minus = get_observable_vector(dataset, params_minus)

            dy_dp = (y_plus - y_minus) / actual_step

            G_abs_dataset[:, j] = dy_dp
            G_rel_dataset[:, j] = np.abs(dy_dp * p_value / scale)

            mean_sens = np.mean(G_rel_dataset[:, j])
            print(f"  {param_name}: sensibilidad media relativa = {mean_sens:.6f}")

        G_abs_rows.append(G_abs_dataset)
        G_rel_rows.append(G_rel_dataset)

        y_exp_all.append(y_exp)
        y_sim_base_all.append(y_base)

        n_S = len(dataset["sugars_profile"])
        sigma_labels.extend(["S"] * n_S + ["E_final"])

    G_abs = np.vstack(G_abs_rows)
    G_rel = np.vstack(G_rel_rows)

    y_exp_all = np.concatenate(y_exp_all)
    y_sim_base_all = np.concatenate(y_sim_base_all)

    return {
        "G_abs": G_abs,
        "G_rel": G_rel,
        "y_exp": y_exp_all,
        "y_sim_base": y_sim_base_all,
        "sigma_labels": sigma_labels,
        "free_params": free_params,
        "params_base_vector": params_base_vector,
    }


def compute_correlation_matrix(G_rel, free_params):
    """
    Matriz de correlación entre columnas de sensibilidad.
    """

    if G_rel.shape[1] == 1:
        C = np.array([[1.0]])
    else:
        C = np.corrcoef(G_rel.T)

    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(C, index=free_params, columns=free_params)


def compute_fim_significance(
    G_abs,
    sigma_labels,
    params_base_dict,
    free_params,
    sigma2_S,
    sigma2_E,
):
    """
    Calcula FIM aproximada y estadísticos de significancia.
    """

    variances = np.array([
        sigma2_S if label == "S" else sigma2_E
        for label in sigma_labels
    ], dtype=float)

    weights = 1.0 / np.maximum(variances, 1e-12)

    # FIM = G.T @ Q @ G
    # Como Q es diagonal, ponderamos filas de G.
    G_weighted = G_abs * np.sqrt(weights[:, None])
    FIM = G_weighted.T @ G_weighted

    # Inversa robusta por pseudoinversa
    FIM_inv = np.linalg.pinv(FIM)

    sigma2_params = np.diag(FIM_inv)
    sigma2_params = np.maximum(sigma2_params, 0.0)

    sigma_params = np.sqrt(sigma2_params)

    param_values = np.array([params_base_dict[p] for p in free_params], dtype=float)

    t_values = np.abs(param_values / np.maximum(sigma_params, 1e-12))

    CI_l = param_values - 2.0 * sigma_params
    CI_u = param_values + 2.0 * sigma_params

    relative_error_95 = (2.0 * sigma_params / np.maximum(np.abs(param_values), 1e-12)) * 100.0

    significance_df = pd.DataFrame({
        "parametro": free_params,
        "valor_estimado": param_values,
        "sigma_parametro": sigma_params,
        "IC95_inf": CI_l,
        "IC95_sup": CI_u,
        "t_value": t_values,
        "error_relativo_95_%": relative_error_95,
    })

    FIM_df = pd.DataFrame(FIM, index=free_params, columns=free_params)

    return FIM_df, significance_df


def build_recommendation_table(
    G_rel,
    corr_df,
    significance_df,
    free_params,
    sensitivity_threshold=0.01,
    correlation_threshold=0.95,
    tvalue_threshold=2.0,
):
    mean_sensitivity = np.mean(G_rel, axis=0)
    max_sensitivity = np.max(G_rel, axis=0)

    rows = []

    for i, param in enumerate(free_params):
        sens_mean = mean_sensitivity[i]
        sens_max = max_sensitivity[i]

        t_value = float(
            significance_df.loc[
                significance_df["parametro"] == param,
                "t_value"
            ].iloc[0]
        )

        high_corr_with = []

        for other_param in free_params:
            if other_param == param:
                continue

            cij = float(corr_df.loc[param, other_param])

            if abs(cij) >= correlation_threshold:
                high_corr_with.append(f"{other_param} ({cij:.3f})")

        low_sensitivity = sens_mean < sensitivity_threshold
        low_tvalue = t_value < tvalue_threshold
        high_corr = len(high_corr_with) > 0

        problems = []

        if low_sensitivity:
            problems.append("baja sensibilidad")

        if low_tvalue:
            problems.append("t-value bajo")

        if high_corr:
            problems.append("alta correlación")

        if low_sensitivity and low_tvalue:
            recommendation = "FIJAR candidato fuerte"
        elif high_corr and (low_sensitivity or low_tvalue):
            recommendation = "FIJAR/REVISAR candidato"
        elif high_corr:
            recommendation = "REVISAR por correlación"
        elif low_sensitivity or low_tvalue:
            recommendation = "REVISAR"
        else:
            recommendation = "ESTIMAR"

        rows.append({
            "parametro": param,
            "sensibilidad_media_relativa": sens_mean,
            "sensibilidad_max_relativa": sens_max,
            "t_value": t_value,
            "correlacion_alta_con": ", ".join(high_corr_with) if high_corr_with else "",
            "problemas_detectados": ", ".join(problems) if problems else "",
            "recomendacion": recommendation,
        })

    return pd.DataFrame(rows)


def plot_mean_sensitivity(recommendation_df):
    plt.figure(figsize=(9, 5))
    plt.bar(
        recommendation_df["parametro"],
        recommendation_df["sensibilidad_media_relativa"]
    )
    plt.axhline(
        SENSITIVITY_THRESHOLD,
        linestyle="--",
        label=f"Umbral sensibilidad = {SENSITIVITY_THRESHOLD}"
    )
    plt.xlabel("Parámetro")
    plt.ylabel("Sensibilidad media relativa")
    plt.title("Sensibilidad media relativa por parámetro")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(corr_df):
    plt.figure(figsize=(7, 6))
    plt.imshow(corr_df.values, vmin=-1, vmax=1)
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title("Matriz de correlación entre sensibilidades")

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            plt.text(
                j,
                i,
                f"{corr_df.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.show()


def plot_t_values(significance_df):
    plt.figure(figsize=(9, 5))
    plt.bar(significance_df["parametro"], significance_df["t_value"])
    plt.axhline(
        TVALUE_THRESHOLD,
        linestyle="--",
        label=f"Umbral t-value = {TVALUE_THRESHOLD}"
    )
    plt.xlabel("Parámetro")
    plt.ylabel("t-value")
    plt.title("Significancia aproximada de parámetros")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    print("\n====================================================")
    print("ANÁLISIS POST-REGRESIÓN POR DIFERENCIAS FINITAS")
    print("Modelo Coleman")
    print("====================================================")

    print("\nOrden de parámetros:")
    for i, p in enumerate(PARAM_ORDER, start=1):
        print(f"  {i}. {p}")

    print("\nParámetros analizados como libres:")
    for p in free_params_for_analysis:
        print(f"  - {p}")

    fixed_params = [p for p in PARAM_ORDER if p not in free_params_for_analysis]

    print("\nParámetros tratados como fijos:")
    for p in fixed_params:
        print(f"  - {p}: {params_base_dict[p]}")

    datasets = build_datasets(paths)

    params_base_vector = params_dict_to_vector(params_base_dict, PARAM_ORDER)

    sigma2_S, sigma2_E = estimate_measurement_variance(
        datasets=datasets,
        params_base_vector=params_base_vector,
    )

    print("\n=== Varianzas estimadas desde residuos ===")
    print(f"sigma2_S       = {sigma2_S:.6f}")
    print(f"sigma2_E_final = {sigma2_E:.6f}")

    sens_result = finite_difference_sensitivity(
        datasets=datasets,
        params_base_dict=params_base_dict,
        param_order=PARAM_ORDER,
        free_params=free_params_for_analysis,
        rel_step=REL_STEP,
    )

    G_abs = sens_result["G_abs"]
    G_rel = sens_result["G_rel"]
    sigma_labels = sens_result["sigma_labels"]

    corr_df = compute_correlation_matrix(
        G_rel=G_rel,
        free_params=free_params_for_analysis,
    )

    FIM_df, significance_df = compute_fim_significance(
        G_abs=G_abs,
        sigma_labels=sigma_labels,
        params_base_dict=params_base_dict,
        free_params=free_params_for_analysis,
        sigma2_S=sigma2_S,
        sigma2_E=sigma2_E,
    )

    recommendation_df = build_recommendation_table(
        G_rel=G_rel,
        corr_df=corr_df,
        significance_df=significance_df,
        free_params=free_params_for_analysis,
        sensitivity_threshold=SENSITIVITY_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        tvalue_threshold=TVALUE_THRESHOLD,
    )

    print("\n=== TABLA DE SIGNIFICANCIA ===")
    print(significance_df.to_string(index=False))

    print("\n=== MATRIZ DE CORRELACIÓN ===")
    print(corr_df.round(4).to_string())

    print("\n=== RECOMENDACIÓN FINAL ===")
    print(recommendation_df.to_string(index=False))

    plot_mean_sensitivity(recommendation_df)
    plot_correlation_matrix(corr_df)
    plot_t_values(significance_df)

    if SAVE_EXCEL:
        with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
            recommendation_df.to_excel(writer, sheet_name="recomendacion", index=False)
            significance_df.to_excel(writer, sheet_name="significancia", index=False)
            corr_df.to_excel(writer, sheet_name="correlacion")
            FIM_df.to_excel(writer, sheet_name="FIM")

        print(f"\nResultados guardados en: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
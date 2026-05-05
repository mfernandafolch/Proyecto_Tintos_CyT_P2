"""
simulacion.py

Funciones para ejecutar simulaciones de fermentación a partir de datos industriales
y analizar los resultados del modelo dinámico modificado.

Modelo dinámico:
    Estados: x = [X, N, S, E]
        X : biomasa viable (g/L)
        N : nitrógeno asimilable (g/L)
        S : azúcar total fermentable (g/L)
        E : etanol (g/L)

    Entradas:
        T(t)    : temperatura (K)
        Nadd(t) : tasa de adición de nitrógeno (g/L/h)

Incluye:
- Extracción de datos procesados desde Excel.
- Construcción de condiciones iniciales y perfiles de entrada.
- Ejecución de simulación con solve_ivp.
- Organización de resultados simulados.
- Gráficos de simulación y comparación con datos experimentales.
"""

import os
import sys

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from procesamiento_datos import process_excel
from modelo_dinamico_v2 import zenteno_ode_variable


# =============================================================================
# 1. Preparación de datos para simulación
# =============================================================================

def data_for_simulation(excel_path: str, t_muestreo: float = 3.0):
    """
    Procesa un archivo Excel industrial y construye los vectores necesarios
    para simular el modelo modificado.

    Returns
    -------
    dict
        Diccionario con:
        - x0
        - t_rel
        - sugars_profile
        - temp_prom
        - Nadd
        - tspan
        - Et_final
        - data_excel
    """

    data_excel = process_excel(path_excel=excel_path, t_muestreo_h=t_muestreo)

    # Condiciones iniciales del modelo modificado: [X, N, S, E]
    x0 = np.array([
        data_excel.init.X0_gL,
        data_excel.init.N0_gL,
        data_excel.init.G0_gL + data_excel.init.F0_gL,
        data_excel.init.E0_gL
    ], dtype=float)

    # Tiempo relativo en horas
    t_rel = np.asarray(data_excel.profiles.t_rel_h, dtype=float)

    # Perfil experimental de azúcar total
    sugars_profile = np.asarray(data_excel.profiles.azucar, dtype=float)

    # Perfil de temperatura en Kelvin
    temp_prom = np.asarray(data_excel.profiles.temp_promedio, dtype=float) + 273.15

    # Perfil de adición de nitrógeno
    Nadd = np.asarray(data_excel.profiles.Nadd_gL, dtype=float)

    # Intervalo de simulación
    tspan = (float(t_rel[0]), float(t_rel[-1]))

    # Etanol final observado
    Et_final = data_excel.init.E_final_obs_gL

    return {
        "x0": x0,
        "t_rel": t_rel,
        "sugars_profile": sugars_profile,
        "temp_prom": temp_prom,
        "Nadd": Nadd,
        "tspan": tspan,
        "Et_final": Et_final,
        "data_excel": data_excel,
    }


# =============================================================================
# 2. Diagnóstico simple de pulsos de nitrógeno
# =============================================================================

def check_N_pulse(Nadd, t_rel=None, verbose=False):
    """
    Detecta si existe algún pulso de nitrógeno en el perfil Nadd.

    Parameters
    ----------
    Nadd : array-like
        Perfil de adición de nitrógeno.
    t_rel : array-like, optional
        Tiempo relativo asociado al perfil.
    verbose : bool
        Si True, imprime los tiempos donde hay adición.

    Returns
    -------
    bool
        True si existe al menos una adición de nitrógeno.
    """

    Nadd = np.asarray(Nadd, dtype=float)

    has_pulse = np.any(Nadd > 0.0)

    if verbose and has_pulse and t_rel is not None:
        t_rel = np.asarray(t_rel, dtype=float)

        for i, value in enumerate(Nadd):
            if value > 0.0:
                print(
                    f"Adición de nitrógeno detectada en "
                    f"t = {t_rel[i]:.2f} h | Nadd = {value:.6f}"
                )

    return bool(has_pulse)


# =============================================================================
# 3. Simulación
# =============================================================================

def simulate_system(x0, t_rel, temp, Nadd, tspan, params_list):
    """
    Ejecuta la simulación del modelo dinámico modificado.

    Estados simulados:
        y[0] = X
        y[1] = N
        y[2] = S
        y[3] = E

    Parameters
    ----------
    x0 : array-like
        Condición inicial [X0, N0, S0, E0].
    t_rel : array-like
        Tiempos de simulación/evaluación en horas.
    temp : array-like
        Perfil de temperatura en K.
    Nadd : array-like
        Perfil de adición de nitrógeno.
    tspan : tuple
        Intervalo de integración.
    params_list : list or array-like
        Parámetros del modelo modificado.

    Returns
    -------
    OdeResult
        Resultado de solve_ivp.
    """

    x0 = np.asarray(x0, dtype=float)
    t_rel = np.asarray(t_rel, dtype=float)
    temp = np.asarray(temp, dtype=float)
    Nadd = np.asarray(Nadd, dtype=float)

    if len(x0) != 4:
        raise ValueError(f"x0 debe tener 4 estados [X,N,S,E], pero tiene {len(x0)}.")

    if not (len(t_rel) == len(temp) == len(Nadd)):
        raise ValueError(
            "t_rel, temp y Nadd deben tener el mismo largo. "
            f"Recibido: len(t_rel)={len(t_rel)}, len(temp)={len(temp)}, len(Nadd)={len(Nadd)}"
        )

    sol = solve_ivp(
        fun=zenteno_ode_variable,
        t_span=tspan,
        y0=x0,
        method="LSODA",
        t_eval=t_rel,
        args=(params_list, t_rel, temp, Nadd)
    )

    if not sol.success:
        print("[WARNING] La simulación no terminó correctamente.")
        print(f"Mensaje solve_ivp: {sol.message}")

    return sol


def simulate_system_from_path(
    excel_path: str,
    params: list,
    t_muestreo: float = 3.0,
    return_info: bool = False
):
    """
    Procesa un archivo Excel y ejecuta la simulación completa.

    Parameters
    ----------
    excel_path : str
        Ruta del archivo Excel.
    params : list
        Parámetros del modelo.
    t_muestreo : float
        Tiempo de muestreo usado en process_excel.
    return_info : bool
        Si True, retorna también el diccionario de datos procesados.

    Returns
    -------
    sol : OdeResult
        Resultado de solve_ivp.

    info : dict, optional
        Diccionario con datos experimentales y entradas usadas.
    """

    info = data_for_simulation(excel_path, t_muestreo=t_muestreo)

    sol = simulate_system(
        x0=info["x0"],
        t_rel=info["t_rel"],
        temp=info["temp_prom"],
        Nadd=info["Nadd"],
        tspan=info["tspan"],
        params_list=params
    )

    if return_info:
        return sol, info

    return sol


# =============================================================================
# 4. Organización de resultados
# =============================================================================

def simulation_to_dict(sol, scale_N_mgL: bool = True):
    """
    Convierte el resultado de solve_ivp en un diccionario ordenado.

    Parameters
    ----------
    sol : OdeResult
        Resultado de solve_ivp.
    scale_N_mgL : bool
        Si True, agrega N en mg/L además de g/L.

    Returns
    -------
    dict
        Diccionario con tiempo y estados simulados.
    """

    if sol.y.shape[0] != 4:
        raise ValueError(
            f"Se esperaban 4 estados [X,N,S,E], pero sol.y tiene {sol.y.shape[0]}."
        )

    t_h = sol.t
    t_d = sol.t / 24.0

    X = sol.y[0, :]
    N = sol.y[1, :]
    S = sol.y[2, :]
    E = sol.y[3, :]

    out = {
        "t_h": t_h,
        "t_d": t_d,
        "X_gL": X,
        "N_gL": N,
        "S_gL": S,
        "E_gL": E,
    }

    if scale_N_mgL:
        out["N_mgL"] = N * 1000.0

    return out


def print_simulation_summary(sol, info=None):
    """
    Imprime un resumen simple de la simulación.
    """

    sim = simulation_to_dict(sol)

    print("Resumen de simulación")
    print("---------------------")
    print(f"solve_ivp success: {sol.success}")
    print(f"Mensaje: {sol.message}")
    print(f"Tiempo inicial: {sim['t_h'][0]:.2f} h")
    print(f"Tiempo final:   {sim['t_h'][-1]:.2f} h")
    print("")
    print("Estados finales simulados:")
    print(f"X final = {sim['X_gL'][-1]:.4f} g/L")
    print(f"N final = {sim['N_gL'][-1]:.6f} g/L ({sim['N_mgL'][-1]:.2f} mg/L)")
    print(f"S final = {sim['S_gL'][-1]:.4f} g/L")
    print(f"E final = {sim['E_gL'][-1]:.4f} g/L")

    if info is not None:
        print("")
        print("Datos experimentales disponibles:")
        if info.get("Et_final") is not None:
            print(f"E final experimental = {info['Et_final']:.4f} g/L")

        if info.get("sugars_profile") is not None:
            sugars_profile = np.asarray(info["sugars_profile"], dtype=float)
            print(f"S inicial experimental = {sugars_profile[0]:.4f} g/L")
            print(f"S final experimental   = {sugars_profile[-1]:.4f} g/L")

        if check_N_pulse(info["Nadd"]):
            print("Hay pulso de nitrógeno en el perfil Nadd.")
        else:
            print("No hay pulso de nitrógeno en el perfil Nadd.")


# =============================================================================
# 5. Gráficos
# =============================================================================

def plot_simulation(res, path=None, scale_N=True):
    """
    Grafica las variables simuladas del modelo modificado.

    Estados:
        X, N, S, E
    """

    sim = simulation_to_dict(res, scale_N_mgL=scale_N)

    title = "Simulación de fermentación"
    if path is not None:
        title = os.path.splitext(os.path.basename(path))[0]

    t_dias = sim["t_d"]

    X = sim["X_gL"]
    N = sim["N_mgL"] if scale_N else sim["N_gL"]
    S = sim["S_gL"]
    E = sim["E_gL"]

    plt.figure(figsize=(8, 5))

    plt.plot(t_dias, X, "-", label="$X$ biomasa (g/L)")
    plt.plot(
        t_dias,
        N,
        "-",
        label="$N$ nitrógeno (mg/L)" if scale_N else "$N$ nitrógeno (g/L)"
    )
    plt.plot(t_dias, S, "-", label="$S$ azúcar total (g/L)")
    plt.plot(t_dias, E, "-", label="$E$ etanol (g/L)")

    plt.title(f"Simulación de fermentación\n{title}")
    plt.ylabel("Concentración")
    plt.xlabel("Tiempo (días)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_simulation_with_data(
    res,
    path=None,
    sugars_profile=None,
    Et_final=None,
    scale_N=True
):
    """
    Grafica la simulación y compara contra datos experimentales de:
    - azúcar total S
    - etanol final E

    Adaptado al modelo modificado con estados [X,N,S,E].
    """

    sim = simulation_to_dict(res, scale_N_mgL=scale_N)

    title = "Simulación de fermentación"
    if path is not None:
        title = os.path.splitext(os.path.basename(path))[0]

    t_dias = sim["t_d"]

    X = sim["X_gL"]
    N_mgL = sim["N_mgL"]
    N_gL = sim["N_gL"]
    S = sim["S_gL"]
    E = sim["E_gL"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 10.5), sharex=True)
    ax1, ax2 = axes

    # -------------------------------------------------------------------------
    # Panel 1: variables principales del ajuste
    # -------------------------------------------------------------------------
    ax1.plot(t_dias, S, "-", linewidth=2, label="$S$ simulado (g/L)")
    ax1.plot(t_dias, E, "-", linewidth=2, label="$E$ simulado (g/L)")

    if scale_N:
        ax1.plot(t_dias, N_mgL, "-", alpha=0.8, label="$N$ simulado (mg/L)")
    else:
        ax1.plot(t_dias, N_gL, "-", alpha=0.8, label="$N$ simulado (g/L)")

    if sugars_profile is not None:
        sugars_profile = np.asarray(sugars_profile, dtype=float)

        if len(sugars_profile) != len(t_dias):
            print(
                f"[WARNING] sugars_profile tiene largo {len(sugars_profile)} "
                f"y la simulación tiene {len(t_dias)} tiempos. No se graficará."
            )
        else:
            ax1.plot(
                t_dias,
                sugars_profile,
                "o",
                markersize=4,
                label="$S$ experimental (g/L)"
            )

    if Et_final is not None:
        ax1.plot(
            t_dias[-1],
            Et_final,
            "s",
            markersize=7,
            label="$E_{final}$ experimental (g/L)"
        )

    ax1.set_title("Azúcar, etanol y nitrógeno")
    ax1.set_ylabel("Concentración")
    ax1.legend()
    ax1.grid(True)

    # -------------------------------------------------------------------------
    # Panel 2: biomasa y nitrógeno en g/L
    # -------------------------------------------------------------------------
    ax2.plot(t_dias, X, "-", linewidth=2, label="$X$ biomasa (g/L)")
    ax2.plot(t_dias, N_gL, "-", linewidth=2, label="$N$ nitrógeno (g/L)")

    ax2.set_title("Biomasa y nitrógeno")
    ax2.set_ylabel("Concentración (g/L)")
    ax2.set_xlabel("Tiempo (días)")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Simulación de fermentación con solve_ivp\n{title}")
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.95), h_pad=2.2)
    plt.show()


def plot_simulation_from_path(excel_path, params, t_muestreo=3.0, scale_N=True):
    """
    Función rápida: procesa, simula y grafica con datos experimentales.
    """

    sol, info = simulate_system_from_path(
        excel_path=excel_path,
        params=params,
        t_muestreo=t_muestreo,
        return_info=True
    )

    print_simulation_summary(sol, info=info)

    plot_simulation_with_data(
        res=sol,
        path=excel_path,
        sugars_profile=info["sugars_profile"],
        Et_final=info["Et_final"],
        scale_N=scale_N
    )

    return sol, info
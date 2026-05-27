"""
simulacion_coleman.py

Funciones para ejecutar simulaciones de fermentación usando el modelo Coleman/Cramer
adaptado a 4 estados.

Estados del modelo:
    x = [X, N, S, E]

donde:
    X : biomasa activa efectiva [g/L]
    N : nitrógeno asimilable [g/L]
    S : azúcares totales [g/L]
    E : etanol [g/L]

Notas:
    - La temperatura se entrega al modelo en °C.
    - El nitrógeno se trabaja en g/L dentro del modelo.
    - Para graficar N, se puede convertir a mg/L multiplicando por 1000.
"""

from procesamiento_datos import process_excel
from modelo_dinamico_coleman import coleman_ode_variable

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os


def data_for_simulation(excel_path: str, t_muestreo=3.0):
    """
    Extrae y prepara los datos necesarios para simular con el modelo Coleman.

    Retorna:
        x0             : condición inicial [X0, N0, S0, E0]
        t_rel          : tiempos relativos [h]
        sugars_profile : perfil experimental de azúcares totales [g/L]
        temp_prom      : perfil de temperatura [°C]
        Nadd           : perfil de adición de nitrógeno [g/L/h]
        tspan          : intervalo de simulación
        Et_final       : etanol final experimental [g/L]
    """

    data_excel = process_excel(
        path_excel=excel_path,
        t_muestreo_h=t_muestreo
    )

    # -------------------------
    # Condición inicial Coleman
    # -------------------------
    X0 = data_excel.init.X0_gL
    N0 = data_excel.init.N0_gL
    S0 = data_excel.init.G0_gL + data_excel.init.F0_gL
    E0 = data_excel.init.E0_gL

    x0 = np.array([X0, N0, S0, E0], dtype=float)

    # -------------------------
    # Perfiles experimentales
    # -------------------------
    t_rel = np.asarray(data_excel.profiles.t_rel_h, dtype=float)

    sugars_profile = np.asarray(data_excel.profiles.azucar, dtype=float)

    # OJO: ahora la temperatura queda en °C.
    # No sumar 273.15.
    temp_prom = np.asarray(data_excel.profiles.temp_promedio, dtype=float)

    Nadd = np.asarray(data_excel.profiles.Nadd_gL, dtype=float)

    tspan = (float(t_rel[0]), float(t_rel[-1]))

    Et_final = data_excel.init.E_final_obs_gL

    return [x0, t_rel, sugars_profile, temp_prom, Nadd, tspan, Et_final]


def check_N_pulse(Nadd, t_rel):
    """
    Revisa si existe al menos una adición de nitrógeno.
    """

    Nadd = np.asarray(Nadd, dtype=float)

    for i in range(len(Nadd)):
        if Nadd[i] > 0.0:
            return True

    return False


def simulate_system(x0, t_rel, temp, Nadd, tspan, params_list):
    """
    Ejecuta solve_ivp usando el wrapper coleman_ode_variable.

    Parámetros:
        x0          : [X0, N0, S0, E0]
        t_rel       : tiempos de evaluación [h]
        temp        : temperatura [°C]
        Nadd        : adición de nitrógeno [g/L/h]
        tspan       : intervalo de simulación
        params_list : [mu0, kd0, betaS0, Kn, Yxn, Yes, Ks]
    """

    sol = solve_ivp(
        fun=coleman_ode_variable,
        t_span=tspan,
        y0=x0,
        method="LSODA",
        t_eval=t_rel,
        args=(params_list, t_rel, temp, Nadd)
    )

    return sol


def simulate_system_from_path(excel_path: str, params: list, t_muestreo=3.0):
    """
    Ejecuta una simulación directamente desde el path del Excel.
    """

    x0, t_rel, sugars_profile, temp_prom, Nadd, tspan, Et_final = data_for_simulation(
        excel_path=excel_path,
        t_muestreo=t_muestreo
    )

    sol = simulate_system(
        x0=x0,
        t_rel=t_rel,
        temp=temp_prom,
        Nadd=Nadd,
        tspan=tspan,
        params_list=params
    )

    return sol


def simulate_system_from_path_with_data(excel_path: str, params: list, t_muestreo=3.0):
    """
    Ejecuta una simulación desde Excel y además retorna los datos experimentales
    útiles para graficar o calcular costo.

    Retorna:
        sol
        data_dict
    """

    x0, t_rel, sugars_profile, temp_prom, Nadd, tspan, Et_final = data_for_simulation(
        excel_path=excel_path,
        t_muestreo=t_muestreo
    )

    sol = simulate_system(
        x0=x0,
        t_rel=t_rel,
        temp=temp_prom,
        Nadd=Nadd,
        tspan=tspan,
        params_list=params
    )

    data_dict = {
        "x0": x0,
        "t_rel": t_rel,
        "sugars_profile": sugars_profile,
        "temp_prom": temp_prom,
        "Nadd": Nadd,
        "tspan": tspan,
        "Et_final": Et_final,
    }

    return sol, data_dict


def plot_simulation(res, path, scale_N=True):
    """
    Grafica las variables simuladas del modelo Coleman.

    Estados:
        y[:, 0] = X
        y[:, 1] = N
        y[:, 2] = S
        y[:, 3] = E
    """

    title = os.path.splitext(os.path.basename(path))[0]

    t = res.t
    t_dias = t / 24.0
    y = res.y.T

    X = y[:, 0]
    N = y[:, 1] * 1000.0 if scale_N else y[:, 1]
    S = y[:, 2]
    E = y[:, 3]

    plt.figure(figsize=(8, 5))

    plt.plot(t_dias, X, "-", label="$X$ (g/L)")
    plt.plot(t_dias, N, "-", label="$N$ (mg/L)" if scale_N else "$N$ (g/L)")
    plt.plot(t_dias, S, "-", label="$S$ (g/L)")
    plt.plot(t_dias, E, "-", label="$E$ (g/L)")

    plt.title(f"Simulación de fermentación con solve_ivp\n{title}")
    plt.ylabel("Concentración")
    plt.xlabel("Tiempo (días)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_simulation_with_data(res, path, sugars_profile=None, Et_final=None, scale_N=True):
    """
    Grafica la simulación Coleman y compara contra:
        - azúcares experimentales totales
        - etanol final experimental
    """

    title = os.path.splitext(os.path.basename(path))[0]

    t = res.t
    t_dias = t / 24.0
    y = res.y.T

    X = y[:, 0]
    N_gL = y[:, 1]
    N_plot = N_gL * 1000.0 if scale_N else N_gL
    S = y[:, 2]
    E = y[:, 3]

    fig, axes = plt.subplots(2, 1, figsize=(9, 10.5), sharex=True)
    ax1, ax2 = axes

    # -------------------------
    # Gráfico 1: variables principales
    # -------------------------
    ax1.plot(
        t_dias,
        S,
        "-",
        linewidth=2,
        label="$S$ simulado (g/L)"
    )

    ax1.plot(
        t_dias,
        E,
        "-",
        linewidth=2,
        label="$E$ simulado (g/L)"
    )

    ax1.plot(
        t_dias,
        N_plot,
        "-",
        label="$N$ simulado (mg/L)" if scale_N else "$N$ simulado (g/L)"
    )

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

    ax1.set_title("Azúcares, etanol y nitrógeno")
    ax1.set_ylabel("Concentración")
    ax1.legend()
    ax1.grid(True)

    # -------------------------
    # Gráfico 2: biomasa y nitrógeno en g/L
    # -------------------------
    ax2.plot(
        t_dias,
        X,
        "k-",
        linewidth=2,
        label="$X$ (g/L)"
    )

    ax2.plot(
        t_dias,
        N_gL,
        "-",
        label="$N$ (g/L)"
    )

    ax2.set_title("Biomasa y nitrógeno")
    ax2.set_ylabel("Concentración (g/L)")
    ax2.set_xlabel("Tiempo (días)")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Simulación de fermentación con modelo Coleman\n{title}")
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.95), h_pad=2.2)

    plt.show()
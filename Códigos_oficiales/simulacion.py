"""
simulacion.py

Funciones para ejecutar simulaciones de fermentación a partir de datos industriales
y analizar los resultados del modelo dinámico.

Incluye:
- Llamada a la función `process_excel` del archivo procesamiento_datos.py para extraer y estructurar 
los datos experimentales desde archivos Excel.
- Construcción de los vectores necesarios para la simulación (condiciones iniciales, perfil temporal, 
temperatura y adición de nitrógeno).
- Función para detectar si existe un pulso de nitrógeno (Nadd) en la fermentación.
- Funciones de diagnóstico para verificar el pulso de nitrógeno, incluyendo el cálculo del 
área total agregada.
- Ejecución de la simulación usando `solve_ivp` y la función `zenteno_ode_variable` definida 
en modelo_dinamico.py.
- Funciones para graficar la evolución de las variables de estado del modelo de fermentación.
"""

from procesamiento_datos import process_excel
# from modelo_dinamico import zenteno_ode_variable, nadd_smooth_from_events, extract_nadd_events
from modelo_dinamico import zenteno_ode_variable

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os

def data_for_simulation(excel_path: str, t_muestreo = 3.0):
    data_excel = process_excel(path_excel=excel_path, t_muestreo_h = t_muestreo)
    
    x0 = np.array([data_excel.init.X0_gL, data_excel.init.N0_gL, 
             data_excel.init.G0_gL, data_excel.init.F0_gL, data_excel.init.E0_gL])
    
    t_rel = data_excel.profiles.t_rel_h
    
    sugars_profile = data_excel.profiles.azucar
   
    temp_prom = data_excel.profiles.temp_promedio + 273.15 # Temp en K
    
    Nadd = data_excel.profiles.Nadd_gL
    tspan = (t_rel[0], t_rel[-1])
    
    Et_final = data_excel.init.E_final_obs_gL
    
    return [x0, t_rel, sugars_profile, temp_prom, Nadd, tspan, Et_final]
    

def check_N_pulse(Nadd: list, t_rel):
    N_pulse = False
    for i in range(len(Nadd)): 
        if Nadd[i] > 0.0:
            # print(f"Adición de nitrógeno en t = {t_rel[i]:.2f} h: Nadd = {Nadd[i]:.3f} g/L*h")
            N_pulse = True
    return N_pulse

"""
# def area_under_nadd_continuous(t_eval, events, duration_h=1.0, k=12.0, n=5000):
#     #Calcula el área bajo Nadd(t) cuando el pulso se define como función continua
#     #(doble sigmoide), independiente del tamaño del paso de tiempo.
#     #Returns
#     #-------
#     #area_gL : float
#     #    N total agregado (g/L)
#     #area_mgL : float
#     #    N total agregado (mg/L)
#     #peak : float
#     #   valor máximo de Nadd (g/L/h)
    
#     t_fine = np.linspace(t_eval[0], t_eval[-1], n)
#     nadd_fine = np.array([nadd_smooth_from_events(t, events, duration_h, k)
#         for t in t_fine])
#     area_gL = float(np.trapz(nadd_fine, t_fine))
#     return area_gL, area_gL * 1000.0, float(np.max(nadd_fine))
"""

def simulate_system(x0, t_rel, temp, Nadd, tspan, params_list):
    
    sol = solve_ivp(fun = zenteno_ode_variable, t_span = tspan,  y0 = x0,  
                   method = 'LSODA', t_eval = t_rel, 
                   args = (params_list, t_rel, temp, Nadd))
    
    return sol

def simulate_system_from_path(excel_path: str, params: list, t_muestreo):
    
    data_excel = process_excel(path_excel=excel_path, t_muestreo_h = t_muestreo)
    
    x0 = np.array([data_excel.init.X0_gL, data_excel.init.N0_gL, 
             data_excel.init.G0_gL, data_excel.init.F0_gL, data_excel.init.E0_gL])
    
    t_rel = data_excel.profiles.t_rel_h
    # print(t_abs)
    temp_prom = data_excel.profiles.temp_promedio + 273.15 # Temp en K
    # print(temp_prom)
    Nadd = data_excel.profiles.Nadd_gL
    tspan = (t_rel[0], t_rel[-1])
    
    # Chequear si hay pulso de nitrógeno
    # if check_N_pulse(Nadd, t_rel):
        # print("Si hay pulso de Nitrógeno")
        # Chequear si está bien aplicado el pulso de nitrógeno (peak y área bajo la curva)
        # events = extract_nadd_events(t_rel, Nadd)
        # area_gL, area_mgL, peak = area_under_nadd_continuous(t_rel, events)
        # print(f"[CHECK Nadd CONT] peak = {peak:.6f} g/L/h | area = {area_gL:.6f} g/L ({area_mgL:.1f} mg/L)")
    # else:
        # print("No hay pulso de Nitrógeno")
        
    # Función "zenteno_ode_variable" tiene toda la lógica de aplicar el peak de nitrógeno 
    # como doble sigmoide si aplica el caso. 
    
    sol = simulate_system(x0=x0, t_rel=t_rel, temp=temp_prom, 
                          Nadd=Nadd, tspan=tspan, params_list=params)
    
    return sol
    
    
def plot_simulation(res, path, scale_N=True):
    """Grafica las variables de una simulación de fermentación.
    Parameters
    ----------
    res : object
        Resultado de solve_ivp (con res.t y res.y)
    path : str
        Path del archivo de datos. Se usa el nombre del archivo como título.
    scale_N : bool
        Si True multiplica N por 1000 para pasar de g/L a mg/L"""

    # Extraer nombre del archivo sin extensión
    title = os.path.splitext(os.path.basename(path))[0]

    # Variables
    t = res.t
    t_dias = t / 24
    y = res.y.T

    X = y[:,0]
    N = y[:,1] * 1000 if scale_N else y[:,1]
    G = y[:,2]
    F = y[:,3]
    E = y[:,4]

    plt.figure(figsize=(8,5))

    plt.plot(t_dias, X, '-', label='$X$ (g/L)')
    plt.plot(t_dias, N, '-', label='$N$ (mg/L)')
    plt.plot(t_dias, G, '-', label='$G$ (g/L)')
    plt.plot(t_dias, F, '-', label='$F$ (g/L)')
    plt.plot(t_dias, E, '-', label='$E$ (g/L)')

    plt.title(f'Simulación de fermentación con solve_ivp\n{title}')
    plt.ylabel('Concentración')
    plt.xlabel('Tiempo (días)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_simulation_with_data(res, path, sugars_profile=None, Et_final=None, scale_N=True):
    """
    Grafica las variables de una simulación de fermentación y compara
    contra datos experimentales de azúcares totales y etanol final.

    Parameters
    ----------
    res : object
        Resultado de solve_ivp (con atributos res.t y res.y)
    path : str
        Path del archivo de datos. Se usa el nombre del archivo como título.
    sugars_profile : array-like, optional
        Perfil experimental de azúcares totales (G+F), evaluado en los mismos
        tiempos de simulación.
    Et_final : float, optional
        Valor experimental del etanol final.
    scale_N : bool, optional
        Si True multiplica N por 1000 para pasar de g/L a mg/L.
    """

    # Extraer nombre del archivo sin extensión
    title = os.path.splitext(os.path.basename(path))[0]

    # Variables de simulación
    t = res.t
    t_dias = t / 24
    y = res.y.T

    X = y[:, 0]
    N = y[:, 1] * 1000 if scale_N else y[:, 1]
    G = y[:, 2]
    F = y[:, 3]
    E = y[:, 4]

    sugars_sim = G + F

    plt.figure(figsize=(9, 6))

    # Curvas simuladas
    plt.plot(t_dias, X, '-', label='$X$ (g/L)')
    plt.plot(t_dias, N, '-', label='$N$ (mg/L)' if scale_N else '$N$ (g/L)')
    #plt.plot(t_dias, G, '-', label='$G$ (g/L)')
    #plt.plot(t_dias, F, '-', label='$F$ (g/L)')
    plt.plot(t_dias, E, '-', label='$E$ (g/L)')

    # Azúcares simulados
    plt.plot(t_dias, sugars_sim, '--', linewidth=2, label='$G+F$ simulado (g/L)')

    # Azúcares experimentales
    if sugars_profile is not None:
        sugars_profile = np.asarray(sugars_profile, dtype=float)

        if len(sugars_profile) != len(t_dias):
            print(
                f"[WARNING] sugars_profile tiene largo {len(sugars_profile)} "
                f"y la simulación tiene {len(t_dias)} tiempos. No se graficará."
            )
        else:
            plt.plot(
                t_dias,
                sugars_profile,
                'o',
                markersize=4,
                label='$G+F$ experimental (g/L)'
            )

    # Etanol final experimental
    if Et_final is not None:
        plt.plot(
            t_dias[-1],
            Et_final,
            's',
            markersize=7,
            label='$E_{final}$ experimental (g/L)'
        )

    plt.title(f'Simulación de fermentación con solve_ivp\n{title}')
    plt.ylabel('Concentración')
    plt.xlabel('Tiempo (días)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
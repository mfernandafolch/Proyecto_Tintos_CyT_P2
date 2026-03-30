# Este código la idea es que saque la info una sola vez al comienzo,
# y después pueda correr varias simulaciones / optimizaciones.

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..",".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from simulacion import data_for_simulation, simulate_system, plot_simulation_with_data
from max_prueba_opt import MODEL_1750, MODEL_1860, MODEL_2264, run_estimation, PARAM_ORDER
from max_prueba_opt import params_dict_to_vector, objective_function, prepare_model_structure
import numpy as np
import time


def format_elapsed(seconds):
    """Devuelve un string legible para tiempos de ejecución."""
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes 
    return f"{minutes} min {rem_seconds:.2f} s" if minutes else f"{rem_seconds:.2f} s"

# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx"
path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx"
# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx"
# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 31.xlsx"
# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx"

# ------------------------------------------------------------
# Extraer datos una sola vez
# ------------------------------------------------------------
data_excel = data_for_simulation(path)

x0 = data_excel[0]
t_rel = data_excel[1]
sugars_profile = data_excel[2]
temp = data_excel[3]
Nadd = data_excel[4]
t_span = data_excel[5]
Et_final = data_excel[6]

# ------------------------------------------------------------
# Elegir estructura de parámetros y método
# ------------------------------------------------------------
model_structure = MODEL_2264
method = "pso"   # "de", "da", "pso", "mealpy_pso"

# ------------------------------------------------------------
# Ejecutar optimización
# ------------------------------------------------------------
opt_start = time.perf_counter()
result, best_params = run_estimation(
    method=method,
    model_structure=model_structure,
    x0=x0,
    t_rel=t_rel,
    temp=temp,
    Nadd=Nadd,
    t_span=t_span,
    sugars_profile=sugars_profile,
    Et_final_exp=Et_final
)
opt_elapsed = time.perf_counter() - opt_start

# ------------------------------------------------------------
# Mostrar resultados
# ------------------------------------------------------------
print("\n=== RESULTADO FINAL ===")

if method in ["de", "da"]:
    print("Mejor costo:", result.fun)

elif method == "pso":
    print("Mejor costo:", result["fun"])

elif method in ["de_ls", "da_ls", "pso_ls"]:
    if method in ["de_ls", "da_ls"]:
        theta_global = result["global_result"].x
    else:
        theta_global = result["global_result"]["x"]

    theta_local = result["x"]

    fixed_params, free_names, _ = prepare_model_structure(model_structure)

    global_cost_obj = objective_function(
        theta_global,
        free_names,
        fixed_params,
        x0,
        t_rel,
        temp,
        Nadd,
        t_span,
        sugars_profile,
        Et_final
    )

    local_cost_obj = objective_function(
        theta_local,
        free_names,
        fixed_params,
        x0,
        t_rel,
        temp,
        Nadd,
        t_span,
        sugars_profile,
        Et_final
    )

    print("Costo etapa global (objective_function):", global_cost_obj)
    print("Costo final refinado (objective_function):", local_cost_obj)
    print("Cost interno de least_squares:", 0.5 * np.sum(result["local_result"].fun**2))

else:
    print("No se reconoce el formato de salida del método.")

print(f"Tiempo total de optimización ({method}): {format_elapsed(opt_elapsed)}")

best_params_list = params_dict_to_vector(best_params, PARAM_ORDER)

print("\nVector ordenado de parámetros:")
for name, value in zip(PARAM_ORDER, best_params_list):
    print(f"{name}: {value}")
print("Número de parámetros:", len(best_params_list))
    
    
res_opt = simulate_system(x0, t_rel, temp, Nadd, t_span, best_params_list)

plot_simulation_with_data(res_opt, path, sugars_profile, Et_final)

# PSO, SCIPY GLOBALES
# ESQUEMA DE REGRESIÓN, PESOS/PONDERAR FUNCIÓN DE COSTOS
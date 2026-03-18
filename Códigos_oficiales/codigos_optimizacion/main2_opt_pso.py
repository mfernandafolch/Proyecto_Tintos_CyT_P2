import os
import sys
import time
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from simulacion import data_for_simulation, simulate_system, plot_simulation_with_data
from pso_opt import (MODEL_1750, MODEL_1860, MODEL_2264, PARAM_ORDER, PSO_CONFIG, run_estimation, params_dict_to_vector,)


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes
    return f"{minutes} min {rem_seconds:.2f} s" if minutes else f"{rem_seconds:.2f} s"


path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx"
# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx"
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

# method = "pso_custom"
# method = "pso_pyswarms"
# method = "pso_mealpy"
method = "pso_pymoo"

# ------------------------------------------------------------
# Configuración única para todos los PSO
# ------------------------------------------------------------
pso_config = PSO_CONFIG.copy()
pso_config["epoch"] = 500
pso_config["pop_size"] = 20
pso_config["w"] = 0.7
pso_config["c1"] = 1.5
pso_config["c2"] = 1.5
pso_config["seed"] = 123
pso_config["verbose"] = True

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
    Et_final_exp=Et_final,
    pso_config=pso_config
)
opt_elapsed = time.perf_counter() - opt_start

# ------------------------------------------------------------
# Mostrar resultados
# ------------------------------------------------------------
print("\n=== RESULTADO FINAL ===")
print("Método:", result["method"])
print("Mejor costo:", result["fun"])
print(f"Tiempo total de optimización ({method}): {format_elapsed(opt_elapsed)}")

best_params_list = params_dict_to_vector(best_params, PARAM_ORDER)

print("\nVector ordenado de parámetros:")
for name, value in zip(PARAM_ORDER, best_params_list):
    print(f"{name}: {value}")
print("Número de parámetros:", len(best_params_list))

# ------------------------------------------------------------
# Simulación final con mejores parámetros
# ------------------------------------------------------------
res_opt = simulate_system(x0, t_rel, temp, Nadd, t_span, best_params_list)
plot_simulation_with_data(res_opt, path, sugars_profile, Et_final)
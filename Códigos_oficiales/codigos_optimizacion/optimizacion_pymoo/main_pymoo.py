import os
import sys
import time
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import data_for_simulation, simulate_system, plot_simulation_with_data
from pymoo_opt import (MODEL_2264,PARAM_ORDER,PSO_CONFIG,run_pymoo_estimation,params_dict_to_vector, plot_pymoo_history)


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes
    return f"{minutes} min {rem_seconds:.2f} s" if minutes else f"{rem_seconds:.2f} s"


paths = [ # Cabernet Sauvignon
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    # Syrah 
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU+VAL+FN estanque 36.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 VAL+STARAQ estanque 56.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU estanque 62.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 25 LOU estanque 30.xlsx",
    # Merlot 
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 Q. AGUA estanque 85.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 24 QAGUA estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 AURORA + STA MARTA estanque 57.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 STA MARTA estanque 62.xlsx",
    # Cabernet Sauvignon 52.400 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 BOLDO estanque 159.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 25 EL BOLDO estanque 133.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 RH+BOLDO estanque 140.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 CONQ+IVALDES estanque 144.xlsx",
]


def build_datasets(paths):
    datasets = []

    for path in paths:
        data_excel = data_for_simulation(path)

        dataset = {
            "path": path,
            "x0": data_excel[0],
            "t_rel": data_excel[1],
            "sugars_profile": data_excel[2],
            "temp": data_excel[3],
            "Nadd": data_excel[4],
            "t_span": data_excel[5],
            "Et_final_exp": data_excel[6],
        }
        datasets.append(dataset)

    return datasets


datasets = build_datasets(paths)

model_structure = MODEL_2264

pso_config = PSO_CONFIG.copy()
pso_config["epoch"] = 1000
pso_config["pop_size"] = 25
pso_config["w"] = 0.7
pso_config["c1"] = 1.5
pso_config["c2"] = 1.5
pso_config["seed"] = 123
pso_config["verbose"] = True
pso_config["relative_gap_threshold"] = 0.01 
# criterio de convergencia: si el gap relativo entre el mejor costo y el promedio de los costos 
# de la población es menor a este umbral, se detiene la optimización.

opt_start = time.perf_counter()
result, best_params = run_pymoo_estimation(
    model_structure=model_structure,
    datasets=datasets,
    pso_config=pso_config
)
opt_elapsed = time.perf_counter() - opt_start

print("\n=== RESULTADO FINAL ===")
print("Método:", result["method"])
print("Mejor costo total:", result["fun"])
print(f"Tiempo total de optimización: {format_elapsed(opt_elapsed)}")

best_params_list = params_dict_to_vector(best_params, PARAM_ORDER)

print("\nVector ordenado de parámetros:")
for name, value in zip(PARAM_ORDER, best_params_list):
    print(f"{name}: {value}")
print("Número de parámetros:", len(best_params_list))

print("\n=== SIMULACIONES FINALES POR DATASET ===")
for i, dataset in enumerate(datasets, start=1):
    print(f"\nDataset {i}: {dataset['path']}")
    res_opt = simulate_system(
        dataset["x0"],
        dataset["t_rel"],
        dataset["temp"],
        dataset["Nadd"],
        dataset["t_span"],
        best_params_list
    )
    plot_simulation_with_data(
        res_opt,
        dataset["path"],
        dataset["sugars_profile"],
        dataset["Et_final_exp"]
    )
plot_pymoo_history(result)
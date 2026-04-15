import os
import sys
import numpy as np


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import simulate_system_from_path, plot_simulation_with_data, data_for_simulation

paths = [# prueba 1: 4, 8, 13, 14
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 25 LOU estanque 30.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 31.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 59.xlsx",
    # prueba 2: 1, 6, 9, 12
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 VAL+STARAQ estanque 56.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 Q. AGUA estanque 85.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 STA MARTA estanque 62.xlsx",
]


# params1 = [0.47047601812319406, 3.0988020709449073, 4.852078904668317, 
#             0.009647, 8.551854, 7.16565, 44.15067, 42.528284, 0.0001, 
#             1.0034902280423204, 6.030732643392028, 
#             1.642634, 
#             0.4995781120314859, 0.313378093771220333]

params2 = [0.09804925636111583, 3.7642227106778177, 5.021829747958857, 
            0.009647, 8.551854, 7.16565, 44.15067, 42.528284, 0.0001, 
            1.0193069327270428, 5.371498936215227, 
            1.642634, 
            0.3581392431405408, 0.48519944839122403]


def compute_validation_cost(sol, sugars_profile, Et_final_exp, penalty=1e12, eps=1e-8):
    y = sol.y.T
    sugars_sim = np.asarray(y[:, 2] + y[:, 3], dtype=float)
    Et_final_sim = float(y[-1, 4])

    sugars_profile = np.asarray(sugars_profile, dtype=float)
    Et_final_exp = float(Et_final_exp)

    if len(sugars_sim) != len(sugars_profile):
        return {
            "sugar_error_mean": penalty,
            "ethanol_error": penalty,
            "objective_total": penalty,
            "Et_final_sim": Et_final_sim,
        }

    if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
        return {
            "sugar_error_mean": penalty,
            "ethanol_error": penalty,
            "objective_total": penalty,
            "Et_final_sim": Et_final_sim,
        }

    sugar_scale = max(np.max(np.abs(sugars_profile)), eps)
    ethanol_scale = max(abs(Et_final_exp), eps)

    sugar_res = (sugars_sim - sugars_profile) / sugar_scale
    etoh_res = (Et_final_sim - Et_final_exp) / ethanol_scale

    sugar_error_mean = float(np.mean(sugar_res ** 2))
    ethanol_error = float(etoh_res ** 2)
    objective_total = float(sugar_error_mean + ethanol_error)

    return {
        "sugar_error_mean": sugar_error_mean,
        "ethanol_error": ethanol_error,
        "objective_total": objective_total,
        "Et_final_sim": Et_final_sim,
    }


soluciones = []
costos = []
datos_validacion = []

for path in paths:
    datos = data_for_simulation(path)
    datos_validacion.append(datos)

    sol = simulate_system_from_path(path, params2)
    soluciones.append(sol)

    costo = compute_validation_cost(
        sol=sol,
        sugars_profile=datos[2],
        Et_final_exp=datos[6]
    )
    costos.append(costo)

print("\n=== COSTO DE VALIDACION POR SET ===")
for i, path in enumerate(paths, start=1):
    print(f"\nSet {i}: {path}")
    print(f"  Costo total J: {costos[i-1]['objective_total']:.6f}")
    print(f"  Error azucar (promedio): {costos[i-1]['sugar_error_mean']:.6f}")
    print(f"  Error etanol final: {costos[i-1]['ethanol_error']:.6f}")
    print(f"  E_final simulado: {costos[i-1]['Et_final_sim']:.4f} g/L")
    print(f"  E_final experimental: {float(datos_validacion[i-1][6]):.4f} g/L")

print("\n=== RESUMEN GLOBAL VALIDACION ===")
print(f"Costo total acumulado: {sum(c['objective_total'] for c in costos):.6f}")
print(f"Costo promedio por set: {np.mean([c['objective_total'] for c in costos]):.6f}")

for i in range(len(paths)):
    plot_simulation_with_data(
        soluciones[i],
        paths[i],
        datos_validacion[i][2],
        datos_validacion[i][6]
    )
    
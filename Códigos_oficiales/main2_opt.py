# Este código la idea es que saque la info una sola vez al comienzo,
# y después pueda correr varias simulaciones / optimizaciones.

from simulacion import data_for_simulation, simulate_system, plot_simulation_with_data
from prueba_opt import MODEL_1750, MODEL_1860, MODEL_2264, run_estimation, PARAM_ORDER

# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx"
# path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx"
path = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx"
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
method = "pso"   # "de", "da", "pso"

# ------------------------------------------------------------
# Ejecutar optimización
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Mostrar resultados
# ------------------------------------------------------------
print("\n=== RESULTADO FINAL ===")
if method in ["de", "da"]:
    print("Mejor costo:", result.fun)
else:
    print("Mejor costo:", result["fun"])

print("Mejores parámetros:")
best_params_list = []
for k, v in best_params.items():
    print(f"{k}: {v}")
    
for i in PARAM_ORDER:
    for k, v in best_params.items():
        if k == i:
            best_params_list.append(v)
    
    
res_opt = simulate_system(x0, t_rel, temp, Nadd, t_span, best_params_list)

plot_simulation_with_data(res_opt, path, sugars_profile, Et_final)

# PSO, SCIPY GLOBALES
# ESQUEMA DE REGRESIÓN, PESOS/PONDERAR FUNCIÓN DE COSTOS
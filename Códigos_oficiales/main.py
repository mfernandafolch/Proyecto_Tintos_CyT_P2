
from simulacion import simulate_system_from_path, plot_simulation, plot_simulation_with_data
from procesamiento_datos import process_excel

paths = [
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 25 KEULE L-30 + BOLDO estanque 149.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 31.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 CONQ+IVALDES estanque 144.xlsx"
]

# parámetros que obtuve en una estimación preliminar.
# params = [5.094427, 5.451988, 2.878541, 2.967765, 429.934043, 279.361563, 123.736979, 278.362199, 0.000100,
#          5.797756, 118.306831, 103.880857, 0.578462, 0.405039]
          
params = [0.313888463467926, 0.6163500508794368, 0.7278951347592207, 0.009647, 8.551854, 7.16565, 
          44.15067, 42.528284, 0.0001, 7.380641111576902, 0.7456962852183697, 1.642634, 
          0.5303679177193666, 0.4372332224714423]


soluciones = []
for path in paths:
    sol = simulate_system_from_path(path, params)
    soluciones.append(sol)

# for i in range(len(paths)):
#     plot_simulation(soluciones[i], paths[i])

data = []
for path in paths:
    data.append(process_excel(path))
    
for i in range(len(paths)):
    plot_simulation_with_data(soluciones[i], paths[i], data[i].profiles.azucar, data[i].init.E_final_obs_gL)
    

# PSO, SCIPY GLOBALES
# ESQUEMA DE REGRESIÓN, PESOS/PONDERAR FUNCIÓN DE COSTOS

from simulacion import simulate_system, plot_simulation

paths = [r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx", 
         r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx",
         r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
         r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 31.xlsx", 
         r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx"]

# parámetros que obtuve en una estimación preliminar.
params = [5.094427, 5.451988, 2.878541, 2.967765, 429.934043, 279.361563, 123.736979, 278.362199, 0.000100,
          5.797756, 118.306831, 103.880857, 0.578462, 0.405039]

soluciones = []
for path in paths:
    sol = simulate_system(path, params)
    soluciones.append(sol)

for i in range(len(paths)):
    plot_simulation(soluciones[i], paths[i])

# PSO, SCIPY GLOBALES
# ESQUEMA DE REGRESIÓN, PESOS/PONDERAR FUNCIÓN DE COSTOS
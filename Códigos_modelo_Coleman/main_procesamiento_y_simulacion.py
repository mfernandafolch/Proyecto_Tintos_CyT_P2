"""
main_procesamiento_y_simulacion.py

Este script realiza el procesamiento de datos experimentales y la simulación 
del modelo de fermentación de Coleman, utilizando los datos de archivos Excel. 
Se generan gráficos comparando las simulaciones con los datos experimentales.
"""


from simulacion_coleman import (
    simulate_system_from_path,
    plot_simulation,
    plot_simulation_with_data
)

from procesamiento_datos import process_excel
import numpy as np


paths = [
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 25 KEULE L-30 + BOLDO estanque 149.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 31.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 CONQ+IVALDES estanque 144.xlsx"
]


# -------------------------------------------------------------------------
# Parámetros modelo Coleman adaptado
# -------------------------------------------------------------------------
# Orden:
# params = [mu0, kd0, betaS0, Kn, Yxn, Yes, Ks]
#
# mu0    : factor multiplicativo de mu_max(T)
# kd0    : factor multiplicativo de k'_d(T)
# betaS0 : factor multiplicativo de betaSmax(T)
# Kn     : constante Monod de nitrógeno [g/L]
# Yxn    : rendimiento biomasa/nitrógeno [gX/gN]
# Yes    : rendimiento etanol/azúcar [gE/gS]
# Ks     : constante Michaelis-Menten de azúcar [g/L]
#
# Si mu0 = kd0 = betaS0 = 1, se usa la dinámica base del paper
# para los parámetros dependientes de temperatura.
# -------------------------------------------------------------------------

params = [
    1.0,              # mu0
    1.0,              # kd0
    1.0,              # betaS0
    np.exp(-4.73),    # Kn
    np.exp(2.5975),   # Yxn, para N0 = 150 mg/L
    np.exp(-0.598),   # Yes
    np.exp(2.33)      # Ks
]


# -------------------------------------------------------------------------
# Simulación
# -------------------------------------------------------------------------

soluciones = []

for path in paths:
    sol = simulate_system_from_path(path, params)
    soluciones.append(sol)


# -------------------------------------------------------------------------
# Datos experimentales
# -------------------------------------------------------------------------

data = []

for path in paths:
    data.append(process_excel(path))


# -------------------------------------------------------------------------
# Gráficos
# -------------------------------------------------------------------------

for i in range(len(paths)):
    plot_simulation_with_data(
        soluciones[i],
        paths[i],
        data[i].profiles.azucar,
        data[i].init.E_final_obs_gL
    )

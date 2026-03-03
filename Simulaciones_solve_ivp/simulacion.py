from procesamiento_datos import process_excel
from modelo_dinamico import zenteno_ode_variable

from scipy.integrate import solve_ivp
import numpy as np


def obtain_info(excel_path: str, t_muestreo = 3.0):
    data_excel = process_excel(path_excel=excel_path, t_muestreo_h = t_muestreo)
    return data_excel

def data_for_simulation(data_class):
    x0 = np.array([data_class.init.X0_gL, data_class.init.N0_gL, 
             data_class.init.G0_gL, data_class.init.F0_gL, data_class.init.E0_gL])
    t_abs = data_class.profiles.t_abs_h
    temp_prom = data_class.profiles.temp_promedio
    Nadd = data_class.profiles.Nadd_gL
    tspan = (t_abs[0], t_abs[-1])
    
    return [x0, t_abs, temp_prom, Nadd, tspan]

def check_N_pulse(Nadd: list, t_abs):
    N_pulse = False
    for i in range(len(Nadd)): 
        if Nadd[i] > 0.0:
            # print(f"Adición de nitrógeno en t = {t_abs[i]:.2f} h: Nadd = {Nadd[i]:.3f} g/L/h")
            N_pulse = True
    return N_pulse

def simulate_system(excel_path: str, params: list):
    data_excel = obtain_info(excel_path)
    init_simulation = data_for_simulation(data_excel) # [0: x0, 1: t_abs, 2: temp_prom, 3: Nadd, 4: tspan]
    x0 = init_simulation[0]
    t_abs = init_simulation[1]
    temp_prom = init_simulation[2]
    Nadd = init_simulation[3]
    tspan = init_simulation[4]
    
    if check_N_pulse(Nadd, t_abs):
        print("Si hay pulso de Nitrógeno")
    else:
        print("No hay pulso de Nitrógeno")
    
    sol = solve_ivp(fun = zenteno_ode_variable, t_span = tspan,  y0 = x0,  
                   method = 'LSODA', t_eval = t_abs, 
                   args = (params, t_abs, temp_prom, Nadd))
    
    return sol
    

paths = [r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx", 
         r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 AGROCAUQ estanque 68.xlsx",
         r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx"]

params = [5.094427, 5.451988, 2.878541, 2.967765, 429.934043, 279.361563, 123.736979, 278.362199, 0.000100,
          5.797756, 118.306831, 103.880857, 0.578462, 0.405039]

soluciones = []
for path in paths:
    sol = simulate_system(path, params)
    soluciones.append(sol)
    
print(soluciones)
    
    


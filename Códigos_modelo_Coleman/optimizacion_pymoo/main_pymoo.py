import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


from simulacion_coleman import (data_for_simulation, simulate_system, plot_simulation_with_data)

from pymoo_opt_coleman import (MODEL_COLEMAN, PARAM_ORDER, PSO_CONFIG, run_pymoo_estimation, params_dict_to_vector, plot_pymoo_history)


# -------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------

def format_elapsed(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes

    if minutes:
        return f"{minutes} min {rem_seconds:.2f} s"

    return f"{rem_seconds:.2f} s"


def format_fixed_decimals(value, decimals=5):
    if np.isscalar(value):
        return f"{float(value):.{decimals}f}"

    arr = np.asarray(value)

    return np.array2string(
        arr,
        separator=", ",
        formatter={"float_kind": lambda x: f"{x:.{decimals}f}"}
    )


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

paths = [
    # Cabernet Sauvignon 100.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",

    # Syrah 100.000 L
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU+VAL+FN estanque 36.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 VAL+STARAQ estanque 56.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU estanque 62.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 25 LOU estanque 30.xlsx",

    # Merlot 100.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 Q. AGUA estanque 85.xlsx",

    # 52.400 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 BOLDO estanque 159.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 25 EL BOLDO estanque 133.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 RH+BOLDO estanque 140.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 24 CONQ+IVALDES estanque 144.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\52.400 L\Data CA 25 LOU estanque 150.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\52.400 L\Data ME 25 Q. AGUA estanque 147.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\52.400 L\Data ME 25 Q. AGUA estanque 171.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\52.400 L\Data SY 24 LOU estanque 152.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\52.400 L\Data SY 25 LOU + VAL estanque 156.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\52.400 L\Data SY 25 LOU estanque 142.xlsx",
]


# -------------------------------------------------------------------------
# Construcción de datasets
# -------------------------------------------------------------------------

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

        print(f"\nDataset construido para: {path}")
        print(f"Condiciones iniciales x0 = [X0, N0, S0, E0]: {format_fixed_decimals(dataset['x0'])}")
        print(f"Cantidad de datos en perfil de azúcares: {len(dataset['sugars_profile'])}")
        print(f"Temperatura mínima: {np.min(dataset['temp']):.2f} °C")
        print(f"Temperatura máxima: {np.max(dataset['temp']):.2f} °C")
        print(f"Etanol final experimental: {dataset['Et_final_exp']:.4f} g/L")

    return datasets


datasets = build_datasets(paths)


# -------------------------------------------------------------------------
# Modelo y configuración PSO
# -------------------------------------------------------------------------

model_structure = MODEL_COLEMAN

pso_config = PSO_CONFIG.copy()

pso_config["epoch"] = 2000
pso_config["pop_size"] = 25
pso_config["w"] = 0.5
pso_config["c1"] = 1.5
pso_config["c2"] = 1.5
pso_config["seed"] = 123
pso_config["verbose"] = True
pso_config["save_history"] = True
pso_config["relative_gap_threshold"] = 0.001


# -------------------------------------------------------------------------
# Optimización
# -------------------------------------------------------------------------

opt_start = time.perf_counter()

result, best_params = run_pymoo_estimation(
    model_structure=model_structure,
    datasets=datasets,
    pso_config=pso_config
)

opt_elapsed = time.perf_counter() - opt_start


# -------------------------------------------------------------------------
# Resultados
# -------------------------------------------------------------------------

print("\n=== RESULTADO FINAL ===")
print("Método:", result["method"])
print("Mejor costo total:", result["fun"])
print(f"Tiempo total de optimización: {format_elapsed(opt_elapsed)}")

best_params_list = params_dict_to_vector(best_params, PARAM_ORDER)

print("\nVector ordenado de parámetros:")
for name, value in zip(PARAM_ORDER, best_params_list):
    print(f"{name}: {value}")

print("Número de parámetros:", len(best_params_list))


# -------------------------------------------------------------------------
# Simulaciones finales por dataset
# -------------------------------------------------------------------------

print("\n=== SIMULACIONES FINALES POR DATASET ===")

for i, dataset in enumerate(datasets, start=1):
    print(f"\nDataset {i}: {dataset['path']}")

    res_opt = simulate_system(
        x0=dataset["x0"],
        t_rel=dataset["t_rel"],
        temp=dataset["temp"],
        Nadd=dataset["Nadd"],
        tspan=dataset["t_span"],
        params_list=best_params_list
    )

    t_dias_data = dataset["t_rel"] / 24.0
    t_dias_sim = res_opt.t / 24.0

    y = res_opt.y.T

    # Modelo Coleman:
    # y[:, 0] = X
    # y[:, 1] = N
    # y[:, 2] = S
    # y[:, 3] = E
    X = y[:, 0]
    N = y[:, 1]
    S = y[:, 2]
    E = y[:, 3]

    S_final_sim = float(S[-1])
    E_final_sim = float(E[-1])

    print(f"Azúcar final simulado: {S_final_sim:.4f} g/L")
    print(f"Etanol final simulado: {E_final_sim:.4f} g/L")
    print(f"Etanol final experimental: {dataset['Et_final_exp']:.4f} g/L")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    title = os.path.splitext(os.path.basename(dataset["path"]))[0]
    ax1.set_title(f"Entrenamiento modelo Coleman\n{title}")

    line_sugar_data, = ax1.plot(
        t_dias_data,
        dataset["sugars_profile"],
        "o",
        label="Azúcar experimental (g/L)"
    )

    dot_etanol_data, = ax1.plot(
        t_dias_data[-1],
        dataset["Et_final_exp"],
        "o",
        label="Etanol final experimental (g/L)"
    )

    line_sugar_sim, = ax1.plot(
        t_dias_sim,
        S,
        "-",
        label="Azúcar simulado S (g/L)"
    )

    line_etanol_sim, = ax1.plot(
        t_dias_sim,
        E,
        "-",
        label="Etanol simulado E (g/L)"
    )

    line_temp, = ax2.plot(
        t_dias_data,
        dataset["temp"],
        "*-",
        alpha=0.3,
        label="Temperatura promedio (°C)"
    )

    ax1.set_xlabel("Tiempo (días)")
    ax1.set_ylabel("Concentración (g/L)")
    ax2.set_ylabel("Temperatura (°C)")

    lines = [
        line_sugar_data,
        line_sugar_sim,
        dot_etanol_data,
        line_etanol_sim,
        line_temp,
    ]

    labels = [line.get_label() for line in lines]

    ax1.legend(lines, labels, loc="center right")
    ax1.grid(True)

    fig.tight_layout()
    plt.show()

    # También puedes usar tu función ya preparada:
    # plot_simulation_with_data(
    #     res=res_opt,
    #     path=dataset["path"],
    #     sugars_profile=dataset["sugars_profile"],
    #     Et_final=dataset["Et_final_exp"]
    # )


# -------------------------------------------------------------------------
# Convergencia
# -------------------------------------------------------------------------

plot_pymoo_history(result)
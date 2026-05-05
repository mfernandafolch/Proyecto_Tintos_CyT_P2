import time
import numpy as np
import matplotlib.pyplot as plt

from simulacion_v2 import data_for_simulation, simulate_system
from pymoo_opt_v2 import (
    MODEL_2264,
    PARAM_ORDER,
    PSO_CONFIG,
    run_pymoo_estimation,
    params_dict_to_vector,
    plot_pymoo_history
)


# =============================================================================
# Utilidades
# =============================================================================

def format_elapsed(seconds):
    minutes = int(seconds // 60)
    rem_seconds = seconds - 60 * minutes
    return f"{minutes} min {rem_seconds:.2f} s" if minutes else f"{rem_seconds:.2f} s"


def format_fixed_decimals(value, decimals=5):
    if np.isscalar(value):
        return f"{float(value):.{decimals}f}"

    arr = np.asarray(value)
    return np.array2string(
        arr,
        separator=", ",
        formatter={"float_kind": lambda x: f"{x:.{decimals}f}"}
    )


def get_dataset_title(path):
    """
    Extrae un título simple desde la ruta del archivo.
    """
    return path.split("\\")[-1].replace(".xlsx", "")


# =============================================================================
# Rutas
# =============================================================================

paths = [
    # Cabernet Sauvignon 100.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 61.xlsx",

    # Syrah 100.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU+VAL+FN estanque 36.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 VAL+STARAQ estanque 56.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 24 LOU estanque 62.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\100.000 L\Data SY 25 LOU estanque 30.xlsx",

    # Merlot 100.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 Q. AGUA estanque 85.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 24 QAGUA estanque 54.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 AURORA + STA MARTA estanque 57.xlsx",
    r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\100.000 L\Data ME 25 STA MARTA estanque 62.xlsx",

    # Carmenere 100.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 31.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 59.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 24 VAL estanque 62.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\100.000 L\Data CA 25 F.N. estanque 68.xlsx",

    # 80.000 L
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\80.000 L\Data CS 25 BOLDO + STA MARTA estanque 41.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\80.000 L\Data CS 25 EL BOLDO (C88 - 89) estanque 50.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\80.000 L\Data CS+MA 24 COL+JMU+IVALDES estanque 41.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CA\80.000 L\Data CA 25 VAL estanque 38.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\80.000 L\Data ME 24 QAGUA estanque 48.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\80.000 L\Data ME 25 Q. AGUA estanque 40.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\ME\80.000 L\Data ME 25 Q. AGUA estanque 45.xlsx",
    # r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\SY\80.000 L\Data SY 24 LOU estanque 41.xlsx",

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


# =============================================================================
# Construcción de datasets
# =============================================================================

def build_dataset_from_data_for_simulation(path):
    """
    Construye un dataset compatible con la optimización.

    Soporta dos formatos posibles de data_for_simulation:

    1) Formato nuevo:
        dict con llaves:
        x0, t_rel, sugars_profile, temp_prom, Nadd, tspan, Et_final

    2) Formato antiguo:
        tupla/lista:
        [x0, t_rel, sugars_profile, temp_prom, Nadd, tspan, Et_final]
    """

    data_excel = data_for_simulation(path)

    if isinstance(data_excel, dict):
        dataset = {
            "path": path,
            "x0": np.asarray(data_excel["x0"], dtype=float),
            "t_rel": np.asarray(data_excel["t_rel"], dtype=float),
            "sugars_profile": np.asarray(data_excel["sugars_profile"], dtype=float),
            "temp": np.asarray(data_excel["temp_prom"], dtype=float),
            "Nadd": np.asarray(data_excel["Nadd"], dtype=float),
            "t_span": data_excel["tspan"],
            "Et_final_exp": float(data_excel["Et_final"]),
        }

    else:
        dataset = {
            "path": path,
            "x0": np.asarray(data_excel[0], dtype=float),
            "t_rel": np.asarray(data_excel[1], dtype=float),
            "sugars_profile": np.asarray(data_excel[2], dtype=float),
            "temp": np.asarray(data_excel[3], dtype=float),
            "Nadd": np.asarray(data_excel[4], dtype=float),
            "t_span": data_excel[5],
            "Et_final_exp": float(data_excel[6]),
        }

    # Validaciones para modelo nuevo [X, N, S, E]
    if len(dataset["x0"]) != 4:
        raise ValueError(
            f"x0 debe tener 4 estados [X, N, S, E], "
            f"pero tiene {len(dataset['x0'])}. Archivo: {path}"
        )

    if not (
        len(dataset["t_rel"])
        == len(dataset["sugars_profile"])
        == len(dataset["temp"])
        == len(dataset["Nadd"])
    ):
        raise ValueError(
            "Los perfiles t_rel, sugars_profile, temp y Nadd deben tener el mismo largo. "
            f"Archivo: {path}\n"
            f"len(t_rel)={len(dataset['t_rel'])}, "
            f"len(sugars_profile)={len(dataset['sugars_profile'])}, "
            f"len(temp)={len(dataset['temp'])}, "
            f"len(Nadd)={len(dataset['Nadd'])}"
        )

    return dataset


def build_datasets(paths):
    datasets = []

    for path in paths:
        dataset = build_dataset_from_data_for_simulation(path)
        datasets.append(dataset)

        print(f"Dataset construido para: {path}")
        print(f"Condiciones iniciales [X0, N0, S0, E0]: {format_fixed_decimals(dataset['x0'])}")
        print(f"Cantidad de datos en el perfil de azúcares: {len(dataset['sugars_profile'])}")
        print(f"Etanol final experimental: {dataset['Et_final_exp']:.5f} g/L")
        print("")

    return datasets


# =============================================================================
# Gráfico de simulación final
# =============================================================================

def plot_final_simulation(dataset, res_opt, title=None):
    """
    Grafica el ajuste final para el modelo modificado.

    Estados del modelo:
        y[0] = X
        y[1] = N
        y[2] = S
        y[3] = E
    """

    if not res_opt.success:
        print("[WARNING] La simulación final no terminó correctamente.")
        print(res_opt.message)

    y = res_opt.y.T

    if y.shape[1] != 4:
        raise ValueError(
            f"El modelo debe retornar 4 estados [X, N, S, E], "
            f"pero retornó {y.shape[1]}."
        )

    X = y[:, 0]
    N = y[:, 1]
    S = y[:, 2]
    E = y[:, 3]

    sugar_sim = S

    t_dias_data = dataset["t_rel"] / 24.0
    t_dias_sim = res_opt.t / 24.0

    if title is None:
        title = f"Entrenamiento de modelo para {get_dataset_title(dataset['path'])}"

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Datos experimentales
    line_sugar_data, = ax1.plot(
        t_dias_data,
        dataset["sugars_profile"],
        "o",
        color="#00F2FE",
        label="Azúcar experimental (g/L)"
    )

    dot_etanol_data, = ax1.plot(
        t_dias_data[-1],
        dataset["Et_final_exp"],
        "o",
        color="r",
        label="Etanol final experimental (g/L)"
    )

    # Simulación
    line_sugar_sim, = ax1.plot(
        t_dias_sim,
        sugar_sim,
        "-",
        color="#00F2FE",
        label="Azúcar simulado (g/L)"
    )

    line_etanol_sim, = ax1.plot(
        t_dias_sim,
        E,
        "-",
        color="r",
        label="Etanol simulado (g/L)"
    )

    # Temperatura
    line_tp, = ax2.plot(
        t_dias_data,
        dataset["temp"] - 273.15,
        "*-",
        alpha=0.3,
        color="m",
        label="Temperatura promedio (°C)"
    )

    ax1.set_title(title)
    ax1.set_xlabel("Tiempo (días)")
    ax1.set_ylabel("Concentración (g/L)")
    ax2.set_ylabel("Temperatura (°C)")

    lines = [
        line_sugar_data,
        line_sugar_sim,
        dot_etanol_data,
        line_etanol_sim,
        line_tp,
    ]

    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")

    ax1.grid(True)
    fig.tight_layout()
    plt.show()

    # Resumen numérico rápido
    print("Resumen simulación final:")
    print(f"X final simulado: {X[-1]:.5f} g/L")
    print(f"N final simulado: {N[-1]:.5f} g/L")
    print(f"S final simulado: {S[-1]:.5f} g/L")
    print(f"E final simulado: {E[-1]:.5f} g/L")
    print(f"S final experimental: {dataset['sugars_profile'][-1]:.5f} g/L")
    print(f"E final experimental: {dataset['Et_final_exp']:.5f} g/L")


# =============================================================================
# Ejecución principal
# =============================================================================

datasets = build_datasets(paths)

model_structure = MODEL_2264

pso_config = PSO_CONFIG.copy()
pso_config["epoch"] = 1000
pso_config["pop_size"] = 25
pso_config["w"] = 0.5
pso_config["c1"] = 1.5
pso_config["c2"] = 1.5
pso_config["seed"] = 123
pso_config["verbose"] = True
pso_config["relative_gap_threshold"] = 0.01

# criterio de convergencia:
# si el gap relativo entre el mejor costo y el promedio de costos de la población
# es menor a este umbral, se detiene la optimización.

opt_start = time.perf_counter()

result, best_params = run_pymoo_estimation(
    model_structure=model_structure,
    datasets=datasets,
    pso_config=pso_config
)

opt_elapsed = time.perf_counter() - opt_start


# =============================================================================
# Resultados finales
# =============================================================================

print("\n=== RESULTADO FINAL ===")
print("Método:", result["method"])
print("Mejor costo total:", result["fun"])
print(f"Tiempo total de optimización: {format_elapsed(opt_elapsed)}")

best_params_list = params_dict_to_vector(best_params, PARAM_ORDER)

print("\nVector ordenado de parámetros:")
for name, value in zip(PARAM_ORDER, best_params_list):
    print(f"{name}: {value}")

print("Número de parámetros:", len(best_params_list))


# =============================================================================
# Simulaciones finales por dataset
# =============================================================================

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

    plot_final_simulation(
        dataset=dataset,
        res_opt=res_opt
    )


# =============================================================================
# Convergencia
# =============================================================================

plot_pymoo_history(result)
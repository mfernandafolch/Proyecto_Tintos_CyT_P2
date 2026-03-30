import os
import sys
import time

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pymoo_opt import MODEL_2264, PSO_CONFIG
from pso_cv_tuning import run_cv_hyperparameter_search


def main():
    paths = [
        r"C:\Users\p-mfolch\OneDrive - Viña Concha y Toro S.A\Escritorio\Archivos\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx",
        r"C:\Users\p-mfolch\OneDrive - Viña Concha y Toro S.A\Escritorio\Archivos\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 25 LOU estanque 31.xlsx",
        r"C:\Users\p-mfolch\OneDrive - Viña Concha y Toro S.A\Escritorio\Archivos\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx",
        r"C:\Users\p-mfolch\OneDrive - Viña Concha y Toro S.A\Escritorio\Archivos\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 25 EL BOLDO estanque 55.xlsx",
        r"C:\Users\p-mfolch\OneDrive - Viña Concha y Toro S.A\Escritorio\Archivos\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 BOLDO estanque 30.xlsx",
    ]

    # c1_options = [1.3, 1.5, 1.7]
    # c2_options = [1.3, 1.5, 1.7]
    # w_options = [0.5, 0.7, 0.9]
    # pop_size_options = [20, 25, 30]
    # epoch_options = [250, 500, 700]

    c1_options = [1.5]
    c2_options = [1.5]
    w_options = [0.5 , 0.7]
    pop_size_options = [25]
    epoch_options = [100]

    base_pso_config = PSO_CONFIG.copy()
    base_pso_config["seed"] = 123
    base_pso_config["verbose"] = False
    base_pso_config["save_history"] = False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CURRENT_DIR, f"resultados_cv_pso_{timestamp}")

    run_cv_hyperparameter_search(
        paths=paths,
        c1_options=c1_options,
        c2_options=c2_options,
        w_options=w_options,
        pop_size_options=pop_size_options,
        epoch_options=epoch_options,
        output_dir=output_dir,
        model_structure=MODEL_2264,
        base_pso_config=base_pso_config,
        t_muestreo=3.0,
        n_workers=4,
        parallel=True,
    )


if __name__ == "__main__":
    main()
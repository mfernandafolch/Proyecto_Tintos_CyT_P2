import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, least_squares
from mealpy import PSO, FloatVar

from simulacion import simulate_system


# ============================================================
# 1. ORDEN FIJO DE PARÁMETROS
# ============================================================

PARAM_ORDER = [
    "mu0",
    "betaG0",
    "betaF0",
    "Kn0",
    "Kg0",
    "Kf0",
    "Kig0",
    "Kie0",
    "Kd0",
    "Yxn",
    "Yxg",
    "Yxf",
    "Yeg",
    "Yef",
]


# ============================================================
# 2. BOUNDS GENERALES
# ============================================================

BOUNDS_DICT = {
    "mu0": (1e-2, 10),
    "betaG0": (1e-2, 10),
    "betaF0": (1e-2, 10),
    "Kn0": (1e-3, 1.0),
    "Kg0": (1e-1, 100.0),
    "Kf0": (1e-1, 100.0),
    "Kig0": (1e-1, 100.0),
    "Kie0": (1e-1, 100.0),
    "Yxn": (1e-1, 10.0),
    "Yxg": (1e-1, 10.0),
    "Yxf": (1e-1, 10.0),
    "Yeg": (1e-1, 10.0),
    "Yef": (1e-1, 10.0),
}


# ============================================================
# 3. MODELOS
# ============================================================

MODEL_1 = {
    "fixed": {},
    "free": {
        "mu0": 0.278601,
        "betaG0": 1.478706,
        "betaF0": 1.792848,
        "Kn0": 0.064151,
        "Kg0": 58.122659,
        "Kf0": 59.222824,
        "Kig0": 71.630423,
        "Kie0": 90.865909,
        "Yxn": 13.617817,
        "Yxg": 1.804689,
        "Yxf": 6.470613,
        "Yeg": 0.735207,
        "Yef": 0.754100,
    }
}

MODEL_1750 = {
    "fixed": {
        "mu0": 0.197200,
        "betaG0": 0.229613,
        "betaF0": 0.248792,
        "Kf0": 7.165650,
        "Kie0": 42.528284,
        "Yeg": 0.451746,
    },
    "free": {
        "Kn0": 0.017264,
        "Kg0": 26.204288,
        "Kig0": 134.167400,
        "Yxn": 35.363591,
        "Yxg": 15.347849,
        "Yxf": 7.769926,
        "Yef": 0.836931,
    }
}

MODEL_1860 = {
    "fixed": {
        "mu0": 0.197200,
        "betaG0": 0.229613,
        "betaF0": 0.248792,
        "Kig0": 44.150670,
        "Kie0": 42.528284,
        "Yxg": 1.393119,
    },
    "free": {
        "Kn0": 0.027142,
        "Kg0": 53.008633,
        "Kf0": 60.118211,
        "Yxn": 69.985750,
        "Yxf": 7.390758,
        "Yeg": 0.995332,
        "Yef": 0.927205,
    }
}

MODEL_2264 = {
    "fixed": {
        "Kn0": 0.009647,
        "Kg0": 8.551854,
        "Kf0": 7.165650,
        "Kig0": 44.150670,
        "Kie0": 42.528284,
        "Yxf": 1.642634,
    },
    "free": {
        "mu0": 0.277041,
        "betaG0": 0.428944,
        "betaF0": 0.516588,
        "Yxn": 28.923223,
        "Yxg": 3.330956,
        "Yeg": 0.780173,
        "Yef": 0.879215,
    }
}


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def prepare_model_structure(model_structure):
    """
    Separa parámetros fijos y libres.
    Además fija Kd0 en 0.0001.
    """
    fixed_params = model_structure["fixed"].copy()
    free_params = model_structure["free"].copy()

    fixed_params["Kd0"] = 0.0001

    if "Kd0" in free_params:
        del free_params["Kd0"]

    free_names = list(free_params.keys())
    theta0 = np.array([free_params[name] for name in free_names], dtype=float)

    return fixed_params, free_names, theta0


def build_full_params(theta, free_names, fixed_params):
    """
    Reconstruye el diccionario completo de parámetros.
    """
    params = fixed_params.copy()
    for name, value in zip(free_names, theta):
        params[name] = float(value)
    return params


def params_dict_to_vector(params_dict, param_order=PARAM_ORDER):
    """
    Convierte diccionario de parámetros a vector ordenado.
    """
    return np.array([params_dict[name] for name in param_order], dtype=float)


def build_bounds_for_free_params(free_names, bounds_dict):
    """
    Construye bounds en el orden correcto.
    """
    bounds = []
    for name in free_names:
        if name not in bounds_dict:
            raise ValueError(f"No hay bounds definidos para '{name}'")
        bounds.append(bounds_dict[name])
    return bounds


def bounds_to_arrays(bounds):
    """
    Convierte lista de bounds [(lb, ub), ...] a dos arrays.
    """
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    return lb, ub


def simulate_from_theta(theta, free_names, fixed_params, x0, t_rel, temp, Nadd, t_span):
    """
    Ejecuta la simulación a partir de theta y devuelve:
    - resultado solve_ivp
    - azúcares simulados
    - etanol final simulado
    """
    params_dict = build_full_params(theta, free_names, fixed_params)
    params_vector = params_dict_to_vector(params_dict, PARAM_ORDER)

    sim = simulate_system(
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        tspan=t_span,
        params_list=params_vector
    )

    y = sim.y.T
    G_sim = np.asarray(y[:, 2], dtype=float)
    F_sim = np.asarray(y[:, 3], dtype=float)
    E_sim = np.asarray(y[:, 4], dtype=float)

    sugars_sim = G_sim + F_sim
    Et_final_sim = float(E_sim[-1])

    return sim, sugars_sim, Et_final_sim


def compute_objective_breakdown(
    theta,
    free_names,
    fixed_params,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    penalty=1e12
):
    """
    Calcula por separado residuos y aportes al costo.

    Returns
    -------
    dict
    """
    try:
        _, sugars_sim, Et_final_sim = simulate_from_theta(
            theta=theta,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span
        )

        sugars_profile = np.asarray(sugars_profile, dtype=float)

        if len(sugars_sim) != len(sugars_profile):
            raise ValueError(
                f"Largo incompatible entre azúcares simulados ({len(sugars_sim)}) "
                f"y perfil experimental ({len(sugars_profile)})"
            )

        if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
            return {
                "sugar_residual_vector": np.full(len(sugars_profile), np.nan),
                "ethanol_residual": np.nan,
                "sugar_error_sum": penalty,
                "sugar_error_mean": penalty,
                "ethanol_error": penalty,
                "objective_total": penalty,
                "sugars_sim": sugars_sim,
                "Et_final_sim": Et_final_sim,
            }

        sugar_res = sugars_sim - sugars_profile
        etoh_res = Et_final_sim - Et_final_exp

        sugar_error_sum = np.sum(sugar_res ** 2)
        sugar_error_mean = sugar_error_sum / len(sugars_sim)
        ethanol_error = etoh_res ** 2
        objective_total = sugar_error_mean + ethanol_error

        return {
            "sugar_residual_vector": sugar_res,
            "ethanol_residual": float(etoh_res),
            "sugar_error_sum": float(sugar_error_sum),
            "sugar_error_mean": float(sugar_error_mean),
            "ethanol_error": float(ethanol_error),
            "objective_total": float(objective_total),
            "sugars_sim": sugars_sim,
            "Et_final_sim": float(Et_final_sim),
        }

    except Exception as e:
        print(f"[WARNING] Falló compute_objective_breakdown con theta={theta}. Error: {e}")
        return {
            "sugar_residual_vector": np.full(len(sugars_profile), np.nan),
            "ethanol_residual": np.nan,
            "sugar_error_sum": penalty,
            "sugar_error_mean": penalty,
            "ethanol_error": penalty,
            "objective_total": penalty,
            "sugars_sim": None,
            "Et_final_sim": np.nan,
        }


def print_iteration_breakdown(
    theta,
    free_names,
    fixed_params,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    prefix=""
):
    """
    Imprime un resumen corto del desglose de la función objetivo
    para una solución theta dada.
    """
    breakdown = compute_objective_breakdown(
        theta=theta,
        free_names=free_names,
        fixed_params=fixed_params,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp
    )

    sugar_res = breakdown["sugar_residual_vector"]

    if sugar_res is None or np.any(~np.isfinite(sugar_res)):
        print(f"{prefix} costo={breakdown['objective_total']:.6f} | breakdown inválido")
        return

    sugar_rmse = np.sqrt(np.mean(sugar_res ** 2))

    print(
        f"{prefix} "
        f"costo={breakdown['objective_total']:.6f} | "
        f"azucar={breakdown['sugar_error_mean']:.6f} | "
        f"etanol={breakdown['ethanol_error']:.6f} | "
        f"res_E={breakdown['ethanol_residual']:.6f} | "
        f"RMSE_az={sugar_rmse:.6f}"
    )


# ============================================================
# 5. FUNCIÓN OBJETIVO ESCALAR
# ============================================================

def objective_function(
    theta,
    free_names,
    fixed_params,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    penalty=1e12
):
    """
    Función objetivo escalar para métodos globales.
    """
    breakdown = compute_objective_breakdown(
        theta=theta,
        free_names=free_names,
        fixed_params=fixed_params,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp,
        penalty=penalty
    )

    return breakdown["objective_total"]


# ============================================================
# 6. FUNCIÓN DE RESIDUOS PARA least_squares
# ============================================================

def residuals_function(
    theta,
    free_names,
    fixed_params,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    penalty=1e6
):
    """
    Función de residuos para least_squares, consistente con objective_function.
    """
    try:
        breakdown = compute_objective_breakdown(
            theta=theta,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp,
            penalty=penalty
        )

        sugar_res = breakdown["sugar_residual_vector"]
        etoh_res = breakdown["ethanol_residual"]

        if np.any(~np.isfinite(sugar_res)) or not np.isfinite(etoh_res):
            return np.full(len(sugars_profile) + 1, penalty, dtype=float)

        n = len(sugar_res)

        # coherente con objective_function:
        # sum((sugar_res/sqrt(n))^2) = sugar_error_mean
        sugar_res_ls = sugar_res / np.sqrt(n)
        etoh_res_ls = np.array([etoh_res], dtype=float)

        residuals = np.concatenate([sugar_res_ls, etoh_res_ls])

        if not np.all(np.isfinite(residuals)):
            return np.full(len(sugars_profile) + 1, penalty, dtype=float)

        return residuals

    except Exception as e:
        print(f"[WARNING] Falló residuals_function con theta={theta}. Error: {e}")
        return np.full(len(sugars_profile) + 1, penalty, dtype=float)


# ============================================================
# 7. REFINAMIENTO LOCAL
# ============================================================

def run_least_squares_refinement(
    theta_init,
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    verbose=2,
    max_nfev=30
):
    """
    Refinamiento local usando least_squares.
    Muestra desglose por evaluación.
    """
    lb, ub = bounds_to_arrays(bounds)

    ls_state = {"eval": 0}

    def residuals_wrapper(theta, *args):
        ls_state["eval"] += 1

        # prefix = f"[LS] Eval {ls_state['eval']:03d} |"
        # print_iteration_breakdown(
        #     theta=theta,
        #     free_names=free_names,
        #     fixed_params=fixed_params,
        #     x0=x0,
        #     t_rel=t_rel,
        #     temp=temp,
        #     Nadd=Nadd,
        #     t_span=t_span,
        #     sugars_profile=sugars_profile,
        #     Et_final_exp=Et_final_exp,
        #     prefix=prefix
        # )

        return residuals_function(theta, *args)

    result = least_squares(
        fun=residuals_wrapper,
        x0=theta_init,
        bounds=(lb, ub),
        args=(
            free_names,
            fixed_params,
            x0,
            t_rel,
            temp,
            Nadd,
            t_span,
            sugars_profile,
            Et_final_exp
        ),
        verbose=verbose,
        max_nfev=max_nfev
    )

    best_theta = result.x
    best_params = build_full_params(best_theta, free_names, fixed_params)

    return result, best_params


# ============================================================
# 8. OPTIMIZADORES GLOBALES
# ============================================================

def run_differential_evolution(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    maxiter=14,
    popsize=8,
    seed=50
):
    """
    DE mostrando avance y desglose por iteración.
    """
    state = {"iter": 0}

    def de_callback(xk, convergence):
        state["iter"] += 1
        prefix = f"[DE] Iter {state['iter']:03d} | conv={convergence:.6f} |"
        print_iteration_breakdown(
            theta=xk,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp,
            prefix=prefix
        )

    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(
            free_names,
            fixed_params,
            x0,
            t_rel,
            temp,
            Nadd,
            t_span,
            sugars_profile,
            Et_final_exp
        ),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        callback=de_callback,
        disp=False
    )

    best_theta = result.x
    best_params = build_full_params(best_theta, free_names, fixed_params)

    return result, best_params


def run_dual_annealing(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    maxiter=60,
    seed=50
):
    """
    DA mostrando avance y desglose por callback.
    """
    da_state = {"iter": 0}

    def da_callback(x, f, context):
        da_state["iter"] += 1
        prefix = f"[DA] Iter {da_state['iter']:03d} | context={context} |"
        print_iteration_breakdown(
            theta=x,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp,
            prefix=prefix
        )
        return False

    result = dual_annealing(
        func=objective_function,
        bounds=bounds,
        args=(
            free_names,
            fixed_params,
            x0,
            t_rel,
            temp,
            Nadd,
            t_span,
            sugars_profile,
            Et_final_exp
        ),
        maxiter=maxiter,
        seed=seed,
        callback=da_callback,
        no_local_search=False
    )

    best_theta = result.x
    best_params = build_full_params(best_theta, free_names, fixed_params)

    return result, best_params


def run_pso(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    n_particles=25,
    n_iter=100,
    w=0.4, # 0.7
    c1=2.05,  # 1.5
    c2=2.05,  # 1.5
    seed=123
):
    rng = np.random.default_rng(seed)

    n_dim = len(free_names)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    positions = rng.uniform(lb, ub, size=(n_particles, n_dim))
    velocities = np.zeros((n_particles, n_dim))

    pbest_positions = positions.copy()
    pbest_scores = np.full(n_particles, np.inf)

    gbest_position = None
    gbest_score = np.inf

    history = []

    for it in range(n_iter):
        for i in range(n_particles):
            score = objective_function(
                theta=positions[i],
                free_names=free_names,
                fixed_params=fixed_params,
                x0=x0,
                t_rel=t_rel,
                temp=temp,
                Nadd=Nadd,
                t_span=t_span,
                sugars_profile=sugars_profile,
                Et_final_exp=Et_final_exp
            )

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

        for i in range(n_particles):
            r1 = rng.random(n_dim)
            r2 = rng.random(n_dim)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

        history.append(gbest_score)

        prefix = f"[PSO] Iter {it+1:03d}/{n_iter} |"
        print_iteration_breakdown(
            theta=gbest_position,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp,
            prefix=prefix
        )

    best_params = build_full_params(gbest_position, free_names, fixed_params)

    result = {
        "x": gbest_position,
        "fun": gbest_score,
        "history": history
    }

    return result, best_params

def run_mealpy_pso(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp,
    epoch=25,
    pop_size=100
    ):
    
    """
    PSO usando MEALPY.
    """

    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    def mealpy_obj(solution):
        theta = np.asarray(solution, dtype=float)
        return objective_function(
            theta=theta,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    problem = {
        "bounds": FloatVar(lb=lb, ub=ub, name="theta"),
        "minmax": "min",
        "obj_func": mealpy_obj,
    }

    model = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, seed=123)

    g_best = model.solve(problem)

    best_theta = np.asarray(g_best.solution, dtype=float)
    best_cost = float(g_best.target.fitness)
    best_params = build_full_params(best_theta, free_names, fixed_params)

    result = {
        "x": best_theta,
        "fun": best_cost,
        "history": getattr(model, "history", None),
        "mealpy_best": g_best,
    }

    return result, best_params


# ============================================================
# 9. MÉTODOS HÍBRIDOS
# ============================================================

def run_de_then_ls(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp
):
    print("\n=== ETAPA GLOBAL: DIFFERENTIAL EVOLUTION ===")
    result_de, _ = run_differential_evolution(
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp
    )

    print("\n=== ETAPA LOCAL: LEAST SQUARES ===")
    result_ls, best_params = run_least_squares_refinement(
        theta_init=result_de.x,
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp,
        verbose=2,
        max_nfev=20
    )

    return {
        "global_result": result_de,
        "local_result": result_ls,
        "x": result_ls.x,
        "fun": 0.5 * np.sum(result_ls.fun**2)
    }, best_params


def run_da_then_ls(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp
):
    print("\n=== ETAPA GLOBAL: DUAL ANNEALING ===")
    result_da, _ = run_dual_annealing(
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp
    )

    print("\n=== ETAPA LOCAL: LEAST SQUARES ===")
    result_ls, best_params = run_least_squares_refinement(
        theta_init=result_da.x,
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp,
        verbose=2,
        max_nfev=20
    )

    return {
        "global_result": result_da,
        "local_result": result_ls,
        "x": result_ls.x,
        "fun": 0.5 * np.sum(result_ls.fun**2)
    }, best_params


def run_pso_then_ls(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp
):
    print("\n=== ETAPA GLOBAL: PSO ===")
    result_pso, _ = run_pso(
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp
    )

    print("\n=== ETAPA LOCAL: LEAST SQUARES ===")
    result_ls, best_params = run_least_squares_refinement(
        theta_init=result_pso["x"],
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp,
        verbose=2,
        max_nfev=20
    )

    return {
        "global_result": result_pso,
        "local_result": result_ls,
        "x": result_ls.x,
        "fun": 0.5 * np.sum(result_ls.fun**2)
    }, best_params

def run_mealpy_pso_then_ls(
    free_names,
    fixed_params,
    bounds,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp
):
    print("\n=== ETAPA GLOBAL: MEALPY PSO ===")
    result_mpso, _ = run_mealpy_pso(
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp
    )

    print("\n=== ETAPA LOCAL: LEAST SQUARES ===")
    result_ls, best_params = run_least_squares_refinement(
        theta_init=result_mpso["x"],
        free_names=free_names,
        fixed_params=fixed_params,
        bounds=bounds,
        x0=x0,
        t_rel=t_rel,
        temp=temp,
        Nadd=Nadd,
        t_span=t_span,
        sugars_profile=sugars_profile,
        Et_final_exp=Et_final_exp,
        verbose=2,
        max_nfev=20
    )

    return {
        "global_result": result_mpso,
        "local_result": result_ls,
        "x": result_ls.x,
        "fun": 0.5 * np.sum(result_ls.fun**2)
    }, best_params

# ============================================================
# 10. FUNCIÓN GENERAL
# ============================================================

def run_estimation(
    method,
    model_structure,
    x0,
    t_rel,
    temp,
    Nadd,
    t_span,
    sugars_profile,
    Et_final_exp
):
    fixed_params, free_names, theta0 = prepare_model_structure(model_structure)
    bounds = build_bounds_for_free_params(free_names, BOUNDS_DICT)

    print("===================================")
    print(f"Método: {method}")
    print("Parámetros libres:", free_names)
    print("Theta inicial:", theta0)
    print("===================================")

    if method == "de":
        return run_differential_evolution(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    elif method == "da":
        return run_dual_annealing(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    elif method == "pso":
        return run_pso(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    elif method == "de_ls":
        return run_de_then_ls(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    elif method == "da_ls":
        return run_da_then_ls(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    elif method == "pso_ls":
        return run_pso_then_ls(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )
        
    elif method == "mealpy_pso":
        return run_mealpy_pso(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    elif method == "mealpy_pso_ls":
        return run_mealpy_pso_then_ls(
            free_names=free_names,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            t_span=t_span,
            sugars_profile=sugars_profile,
            Et_final_exp=Et_final_exp
        )

    else:
        raise ValueError("Método no reconocido. Usa 'de', 'da', 'pso', 'mealpy_pso', "
        "'de_ls', 'da_ls', 'pso_ls' o 'mealpy_pso_ls'.")
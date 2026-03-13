import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, least_squares

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
    Ejecuta la simulación a partir de theta y devuelve azúcares simulados
    y etanol final simulado.
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

        if len(sugars_sim) != len(sugars_profile):
            raise ValueError(
                f"Largo incompatible entre azúcares simulados ({len(sugars_sim)}) "
                f"y perfil experimental ({len(sugars_profile)})"
            )

        if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
            return penalty
        
        sugar_res = (sugars_sim - sugars_profile)
        etoh_res = (Et_final_sim - Et_final_exp)
        
        sugar_error = np.sum(sugar_res ** 2)
        etoh_error = etoh_res ** 2 # Solo un término para etanol final

        J = sugar_error / len(sugars_sim) + etoh_error

        if not np.isfinite(J):
            return penalty

        return float(J)

    except Exception as e:
        print(f"[WARNING] Falló simulación con theta={theta}. Error: {e}")
        return penalty


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

        if len(sugars_sim) != len(sugars_profile):
            raise ValueError(
                f"Largo incompatible entre azúcares simulados ({len(sugars_sim)}) "
                f"y perfil experimental ({len(sugars_profile)})"
            )

        if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
            return np.full(len(sugars_profile) + 1, penalty, dtype=float)

        n = len(sugars_sim)

        # Importante: dividir por sqrt(n) para que la suma de cuadrados
        # reproduzca el promedio usado en objective_function
        sugar_res = (sugars_sim - sugars_profile) / np.sqrt(n)
        etoh_res = np.array([Et_final_sim - Et_final_exp], dtype=float)

        residuals = np.concatenate([sugar_res, etoh_res])

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
    """
    lb, ub = bounds_to_arrays(bounds)

    result = least_squares(
        fun=residuals_function,
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
    maxiter=12,
    popsize=6,
    seed=50
):
    """
    DE con exigencia reducida y mostrando avance.
    """
    def de_callback(xk, convergence):
        score = objective_function(
            xk,
            free_names,
            fixed_params,
            x0,
            t_rel,
            temp,
            Nadd,
            t_span,
            sugars_profile,
            Et_final_exp
        )
        print(f"[DE] Mejor costo actual: {score:.6f} | convergence={convergence:.6f}")

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
    maxiter=30,
    seed=50
):
    """
    DA con exigencia reducida y mostrando avance por callback.
    """
    da_state = {"iter": 0}

    def da_callback(x, f, context):
        da_state["iter"] += 1
        print(f"[DA] Callback {da_state['iter']} - costo={f:.6f} - context={context}")
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
    n_particles=15,
    n_iter=100,
    w=0.7,
    c1=1.5,
    c2=1.5,
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
        print(f"Iteración PSO {it+1}/{n_iter} - Mejor costo: {gbest_score:.6f}")

    best_params = build_full_params(gbest_position, free_names, fixed_params)

    result = {
        "x": gbest_position,
        "fun": gbest_score,
        "history": history
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

    else:
        raise ValueError("Método no reconocido. Usa 'de', 'da', 'pso', 'de_ls', 'da_ls' o 'pso_ls'.")
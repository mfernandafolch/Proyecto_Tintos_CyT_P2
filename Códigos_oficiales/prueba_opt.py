import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

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


# ============================================================
# 5. FUNCIÓN OBJETIVO
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
    Función objetivo escalar.

    Minimiza:
    1) error cuadrático del perfil de azúcares totales
    2) error cuadrático del etanol final
    """
    try:
        # Diccionario completo
        params_dict = build_full_params(theta, free_names, fixed_params)

        # Vector ordenado, que es lo que probablemente espera simulate_system
        params_vector = params_dict_to_vector(params_dict, PARAM_ORDER)

        sim = simulate_system(
            x0=x0,
            t_rel=t_rel,
            temp=temp,
            Nadd=Nadd,
            tspan=t_span,
            params_list=params_vector
        )

        # Se asume que sim es salida tipo solve_ivp
        y = sim.y.T

        G_sim = np.asarray(y[:, 2], dtype=float)
        F_sim = np.asarray(y[:, 3], dtype=float)
        E_sim = np.asarray(y[:, 4], dtype=float)

        sugars_sim = G_sim + F_sim
        Et_final_sim = float(E_sim[-1])

        # Chequeos básicos
        if len(sugars_sim) != len(sugars_profile):
            raise ValueError(
                f"Largo incompatible entre azúcares simulados ({len(sugars_sim)}) "
                f"y perfil experimental ({len(sugars_profile)})"
            )

        if not (np.all(np.isfinite(sugars_sim)) and np.isfinite(Et_final_sim)):
            return penalty

        sugar_error = np.sum((sugars_sim - sugars_profile) ** 2)
        etoh_error = (Et_final_sim - Et_final_exp) ** 2

        n_obs_sugar = len(sugars_sim)
        n_obs_etoh = 1

        J = 1/n_obs_sugar * sugar_error + 1/n_obs_etoh * etoh_error

        if not np.isfinite(J):
            return penalty

        return float(J)

    except Exception as e:
        print(f"[WARNING] Falló simulación con theta={theta}. Error: {e}")
        return penalty


# ============================================================
# 6. OPTIMIZADORES
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
    maxiter=15,
    popsize=7,
    seed=50
):
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
        polish=False
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
    maxiter=15,
    seed=50
):
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
        seed=seed
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
    n_iter=200,
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
# 7. FUNCIÓN GENERAL
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

    else:
        raise ValueError("Método no reconocido. Usa 'de', 'da' o 'pso'.")
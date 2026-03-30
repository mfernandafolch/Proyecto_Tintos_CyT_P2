import numpy as np
from scipy.optimize import differential_evolution, dual_annealing
from mealpy import PSO, FloatVar

from simulacion import simulate_system


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


def prepare_model_structure(model_structure):
    fixed_params = model_structure["fixed"].copy()
    free_params = model_structure["free"].copy()

    fixed_params["Kd0"] = 0.0001

    if "Kd0" in free_params:
        del free_params["Kd0"]

    free_names = list(free_params.keys())
    theta0 = np.array([free_params[name] for name in free_names], dtype=float)

    return fixed_params, free_names, theta0


def build_full_params(theta, free_names, fixed_params):
    params = fixed_params.copy()
    for name, value in zip(free_names, theta):
        params[name] = float(value)
    return params


def params_dict_to_vector(params_dict, param_order=PARAM_ORDER):
    return np.array([params_dict[name] for name in param_order], dtype=float)


def build_bounds_for_free_params(free_names, bounds_dict):
    bounds = []
    for name in free_names:
        if name not in bounds_dict:
            raise ValueError(f"No hay bounds definidos para '{name}'")
        bounds.append(bounds_dict[name])
    return bounds


def simulate_from_theta(theta, free_names, fixed_params, x0, t_rel, temp, Nadd, t_span):
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
    penalty=1e12,
    eps=1e-8
):
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
                "sugar_scale": np.nan,
                "ethanol_scale": np.nan,
            }

        sugar_scale = max(np.max(np.abs(sugars_profile)), eps)
        ethanol_scale = max(abs(Et_final_exp), eps)

        sugar_res = (sugars_sim - sugars_profile) / sugar_scale
        etoh_res = (Et_final_sim - Et_final_exp) / ethanol_scale

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
            "sugar_scale": float(sugar_scale),
            "ethanol_scale": float(ethanol_scale),
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
            "sugar_scale": np.nan,
            "ethanol_scale": np.nan,
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
        f"RMSE_az_norm={sugar_rmse:.6f}"
    )


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
    maxiter=50,
    popsize=10,
    seed=50
):
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
    maxiter=50,
    seed=50
):
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
    n_iter=500,
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
    epoch=10000,
    pop_size=25,
    w=0.7,
    c1=1.5,
    c2=1.5,
    seed=123
):
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    state = {"eval": 0, "best": np.inf, "best_theta": None}

    def mealpy_obj(solution):
        theta = np.asarray(solution, dtype=float)
        score = objective_function(
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

        state["eval"] += 1
        if score < state["best"]:
            state["best"] = score
            state["best_theta"] = theta.copy()

            # prefix = f"[MEALPY_PSO] Eval {state['eval']:04d} |"
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

        return score

    problem = {
        "bounds": FloatVar(lb=lb, ub=ub, name="theta"),
        "minmax": "min",
        "obj_func": mealpy_obj,
    }

    model = PSO.OriginalPSO(pop_size=pop_size, c1=c1, c2=c2, w=w)
    # model = PSO.HPSO_TVAC(epoch=epoch, pop_size=pop_size, seed=seed)
    
    termination = {
        "max_epoch": epoch,
        "max_early_stop": 50,
        "epsilon": 1e-7
    }

    
    g_best = model.solve(problem, seed=seed, termination=termination)

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

    else:
        raise ValueError("Método no reconocido. Usa 'de', 'da', 'pso' o 'mealpy_pso'.")
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO as PymooPSO
from pymoo.optimize import minimize
from pymoo.core.termination import Termination
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from simulacion import simulate_system


PARAM_ORDER = [
    "mu0", "betaG0", "betaF0", "Kn0", "Kg0", "Kf0", "Kig0",
    "Kie0", "Kd0", "Yxn", "Yxg", "Yxf", "Yeg", "Yef"
]

BOUNDS_DICT = {
    "mu0": (1e-2, 1.0),
    "betaG0": (1e-2, 10.0),
    "betaF0": (1e-2, 10.0),
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

PSO_CONFIG = {
    "epoch": 1000,
    "pop_size": 25,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
    "seed": 123,
    "verbose": True,
    "save_history": True,
    "relative_gap_threshold": 0.01,
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
        "Yxn": 3.000000,
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


def get_pso_config(custom_config=None):
    config = PSO_CONFIG.copy()
    if custom_config is not None:
        config.update(custom_config)
    return config


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

    except Exception:
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


def objective_function_multi(
    theta,
    free_names,
    fixed_params,
    datasets,
    penalty=1e12
):
    total_cost = 0.0

    for data in datasets:
        total_cost += objective_function(
            theta=theta,
            free_names=free_names,
            fixed_params=fixed_params,
            x0=data["x0"],
            t_rel=data["t_rel"],
            temp=data["temp"],
            Nadd=data["Nadd"],
            t_span=data["t_span"],
            sugars_profile=data["sugars_profile"],
            Et_final_exp=data["Et_final_exp"],
            penalty=penalty
        )

    return float(total_cost)


def objective_function_swarm_multi(
    swarm,
    free_names,
    fixed_params,
    datasets,
    penalty=1e12
):
    swarm = np.atleast_2d(swarm)
    costs = np.empty(swarm.shape[0], dtype=float)

    for i, theta in enumerate(swarm):
        costs[i] = objective_function_multi(
            theta=theta,
            free_names=free_names,
            fixed_params=fixed_params,
            datasets=datasets,
            penalty=penalty
        )

    return costs


class FermentationPymooProblem(Problem):
    def __init__(self, free_names, fixed_params, datasets, xl, xu):
        super().__init__(n_var=len(free_names), n_obj=1, xl=xl, xu=xu)
        self.free_names = free_names
        self.fixed_params = fixed_params
        self.datasets = datasets

    def _evaluate(self, X, out, *args, **kwargs):
        F = objective_function_swarm_multi(
            swarm=X,
            free_names=self.free_names,
            fixed_params=self.fixed_params,
            datasets=self.datasets
        )
        out["F"] = F.reshape(-1, 1)

class RelativeGapTermination(Termination):
    def __init__(self, threshold=0.05, eps=1e-12):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        self.last_f_avg = None
        self.last_f_min = None
        self.last_ratio = None
        self.stop_iteration = None
        self.stop_reason = None

    def _update(self, algorithm):
        F = algorithm.pop.get("F")

        if F is None or len(F) == 0:
            return 0.0

        F = np.asarray(F, dtype=float).reshape(-1)

        f_avg = float(np.mean(F))
        f_min = float(np.min(F))

        denom = max(abs(f_min), self.eps)
        ratio = (f_avg - f_min) / denom

        self.last_f_avg = f_avg
        self.last_f_min = f_min
        self.last_ratio = ratio

        if ratio < self.threshold:
            self.stop_iteration = algorithm.n_gen
            self.stop_reason = "relative_gap"
            return 1.0

        return 0.0

def run_pymoo_estimation(
    model_structure,
    datasets,
    pso_config=None
):
    fixed_params, free_names, _ = prepare_model_structure(model_structure)
    bounds = build_bounds_for_free_params(free_names, BOUNDS_DICT)
    config = get_pso_config(pso_config)

    epoch = config["epoch"]
    pop_size = config["pop_size"]
    w = config["w"]
    c1 = config["c1"]
    c2 = config["c2"]
    seed = config["seed"]
    verbose = config["verbose"]
    save_history = config.get("save_history", False)
    relative_gap_threshold = config.get("relative_gap_threshold", 0.05)

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    problem = FermentationPymooProblem(
        free_names=free_names,
        fixed_params=fixed_params,
        datasets=datasets,
        xl=lb,
        xu=ub
    )

    algorithm = PymooPSO(
        pop_size=pop_size,
        w=w,
        c1=c1,
        c2=c2,
        adaptive=False
    )

    relative_gap_termination = RelativeGapTermination(
        threshold=relative_gap_threshold
    )

    max_gen_termination = get_termination("n_gen", epoch)

    termination = TerminationCollection(
        relative_gap_termination,
        max_gen_termination
    )

    pymoo_result = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=seed,
        verbose=verbose,
        save_history=save_history,
    )

    best_theta = np.asarray(pymoo_result.X, dtype=float)
    best_cost = float(np.atleast_1d(pymoo_result.F)[0])

    history_f_min = []
    history_f_avg = []

    if save_history and getattr(pymoo_result, "history", None):
        for algo in pymoo_result.history:
            F = algo.pop.get("F")
            F = np.asarray(F, dtype=float).reshape(-1)
            history_f_min.append(float(np.min(F)))
            history_f_avg.append(float(np.mean(F)))

    best_params = build_full_params(best_theta, free_names, fixed_params)

    if relative_gap_termination.stop_iteration is None:
        stop_iteration = pymoo_result.algorithm.n_gen
        stop_reason = "max_epoch"
    else:
        stop_iteration = relative_gap_termination.stop_iteration
        stop_reason = relative_gap_termination.stop_reason

    final_ratio = relative_gap_termination.last_ratio
    final_f_avg = relative_gap_termination.last_f_avg
    final_f_min = relative_gap_termination.last_f_min

    if final_ratio is None:
        F = pymoo_result.pop.get("F")
        F = np.asarray(F, dtype=float).reshape(-1)

        final_f_avg = float(np.mean(F))
        final_f_min = float(np.min(F))

        denom = max(abs(final_f_min), 1e-12)
        final_ratio = (final_f_avg - final_f_min) / denom

    print(f"Motivo de término: {stop_reason}")
    print(f"Iteración de término: {stop_iteration}")
    print(f"Valor final de (f_avg - f_min)/f_min: {final_ratio:.6f}")

    result = {
        "x": best_theta,
        "fun": best_cost,
        "history": {
            "f_min": history_f_min,
            "f_avg": history_f_avg,
        },
        "method": "pso_pymoo",
        "raw_result": pymoo_result,
        "termination_info": {
            "stop_reason": stop_reason,
            "stop_iteration": stop_iteration,
            "final_ratio": final_ratio,
            "final_f_avg": final_f_avg,
            "final_f_min": final_f_min,
            "threshold": relative_gap_threshold,
            "max_epoch": epoch,
        }
    }

    return result, best_params


def plot_pymoo_history(result, title="Convergencia del PSO con pymoo"):
    history = result.get("history", None)
    if history is None:
        return

    f_min = history.get("f_min", [])
    f_avg = history.get("f_avg", [])

    if len(f_min) == 0:
        return

    generations = np.arange(1, len(f_min) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(generations, f_min, label="f_min")
    if len(f_avg) == len(f_min):
        plt.plot(generations, f_avg, label="f_avg")

    plt.xlabel("Iteración")
    plt.ylabel("Valor objetivo")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
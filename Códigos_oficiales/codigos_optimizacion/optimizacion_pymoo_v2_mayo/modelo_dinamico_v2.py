"""
modelo_dinamico.py

Construcción del modelo dinámico (EDOs) de fermentación tipo Zenteno modificado y su evaluación con entradas variables.

Incluye:
- Utilidades numéricas para estabilidad y robustez: safe_div, safe_exp, clamp, _real_pos.
- Sistema de EDOs del modelo Zenteno (estados x = [X, N, S, E]).
- Suavizado de la adición de nitrógeno Nadd mediante un pulso continuo (doble sigmoide) de duración fija
  (por defecto 1 hora), independiente del tiempo de muestreo de los datos.
- Función zenteno_ode_variable: wrapper compatible con solve_ivp que:
    (i) toma T desde una grilla temporal (piecewise constante),
    (ii) calcula Nadd(t) como función continua a partir de eventos (spikes) detectados en la grilla,
    (iii) llama a zenteno_model_mod para retornar dx/dt.
"""


import numpy as np

# -------------------------------- Utilidades numéricas --------------------------------
EPS = 1e-9
BIG = 1e6  # techo de seguridad para estados y tasas


def safe_div(a, b, eps=EPS):
    return a / (b + eps)


def safe_exp(x, lo=-50.0, hi=50.0):
    """exp con saturación del exponente para evitar overflow/underflow extremo."""
    return np.exp(np.clip(x, lo, hi))


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def _real_pos(z):
    """Parte real y clamp a >= 0 (evita ComplexWarning y negativos numéricos)."""
    r = float(np.real(z))
    return r if r > 0.0 else 0.0

# -------- Funciones para aplicar sigmoide y suavizar la adición de Nitrógeno --------

def _sigmoid(z):
    # estable numéricamente
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

def smooth_pulse_sigmoid(t, t0, duration_h, rate, k=12.0):
    """Pulso suave (doble sigmoide) que aproxima un pulso rectangular:
    - comienza cerca de t0
    - termina cerca de t0 + duration_h
    - altura ~ rate (g/L/h)"""
    t1 = t0 + duration_h
    return rate * (_sigmoid(k*(t - t0)) - _sigmoid(k*(t - t1)))

def extract_nadd_events(t_eval, Nadd_grid, eps=1e-12):
    """Extrae eventos (t0, rate) desde Nadd_grid.
    Interpreta cada valor >0 como: 'a partir de t0 comienza una adición"""
    events = []
    for ti, ni in zip(t_eval, Nadd_grid):
        ni = float(ni)
        if ni > eps:
            events.append((float(ti), ni))
    return events

def nadd_smooth_from_events(t, events, duration_h=1.0, k=12.0):
    """Evalúa Nadd(t) suave como suma de pulsos de DURACIÓN FIJA (1 hora),
    independiente del t_muestreo."""
    out = 0.0
    w = 4.0 / max(k, 1e-12)  # ancho típico de transición sigmoide

    for t0, rate in events:
        # si estamos lejos del pulso, aporta ~0 (ahorra cómputo)
        if t < (t0 - 6*w) or t > (t0 + duration_h + 6*w):
            continue
        out += smooth_pulse_sigmoid(t, t0, duration_h, rate, k=k)

    return float(out)

# -------------------------------- Modelo dinámico --------------------------------
def zenteno_model_mod(t, x, u, p, apply_nadd_in_model=True):
    """
    Modelo de cinética de fermentación modificado.
    Estados: x = [X, N, S, E] (g/L)
    Entradas: u = [T (K), Nadd (g/L/h)]
    Parámetros:
        p = [mu0, betaS0, Kn0, Ks0, Kie0, Kd0, Yxn, Yxs, Yes]
    """

    if len(p) != 9:
        raise ValueError(f"Se esperaban 9 parámetros, pero se recibieron {len(p)}.")

    # Entradas
    T = float(u[0])     # K
    Nadd = float(u[1])  # g/L/h

    # Estados
    X = _real_pos(x[0])
    N = _real_pos(x[1])
    S = _real_pos(x[2])
    E = _real_pos(x[3])

    # Limitar T a rango físico razonable
    T = clamp(T, 273.15, 333.15)

    # Parámetros positivos
    vals = [max(float(pi), EPS) for pi in p]
    (mu0, betaS0, Kn0, Ks0, Kie0, Kd0, Yxn, Yxs, Yes) = vals

    # Constantes
    Cde  = 0.0415
    Etd  = 130000.0
    R    = 8.314
    Eac  = 59453.0
    Eafe = 11000.0
    Eak  = 46055.0
    Eam  = 37681.0
    m0   = 0.01  # fijo para evitar problemas de identificabilidad

    # Factores de Arrhenius
    A_mu   = safe_exp(Eac  * (T - 300.00) / (300.00 * R * T))
    A_beta = safe_exp(Eafe * (T - 296.15) / (296.15 * R * T))
    A_k    = safe_exp(Eak  * (T - 293.15) / (293.15 * R * T))
    A_m    = safe_exp(Eam  * (T - 293.30) / (293.30 * R * T))

    # Parámetros dependientes de temperatura
    mu_max    = mu0    * A_mu
    betaS_max = betaS0 * A_beta
    Kn        = Kn0    * A_k
    Ks        = Ks0    * A_k
    Kie       = Kie0   * A_k
    m         = m0     * A_m

    # Tasas
    mu = mu_max * safe_div(N, N + Kn)

    beta_S = (
        betaS_max
        * safe_div(S, S + Ks)
        * safe_div(Kie, E + Kie)
    )

    # Temperatura umbral de muerte térmica
    E_cap = clamp(E, 0.0, 200.0)

    Td = (
        -0.0001 * E_cap**3
        + 0.0049 * E_cap**2
        - 0.1279 * E_cap
        + 315.89
    )
    Td = clamp(Td, 273.15, 333.15)

    # Decaimiento celular
    if T >= Td:
        exponent = (
            Cde * E_cap
            + safe_div(Etd * (T - 305.65), 305.65 * R * T)
        )
        Kd = Kd0 * safe_exp(exponent, lo=-50.0, hi=50.0)
    else:
        Kd = 0.0

    # EDOs
    dX = (mu - Kd) * X

    dN = -(mu / max(Yxn, EPS)) * X
    if apply_nadd_in_model:
        dN += Nadd

    dS = -(
        (mu / max(Yxs, EPS))
        + (beta_S / max(Yes, EPS))
        + m
    ) * X

    dE = beta_S * X

    # Evitar consumo negativo de azúcar cuando S ya está agotado
    if S <= EPS and dS < 0:
        dS = 0.0

    # Acotar derivadas
    dX = float(clamp(dX, -BIG, BIG))
    dN = float(clamp(dN, -BIG, BIG))
    dS = float(clamp(dS, -BIG, BIG))
    dE = float(clamp(dE, -BIG, BIG))

    return np.array([dX, dN, dS, dE], dtype=float)


def zenteno_ode_variable(t, x, params, t_eval, T_grid, Nadd_grid):
    idx = np.searchsorted(t_eval, t, side="right") - 1
    idx = int(np.clip(idx, 0, len(T_grid)-1))  # seguridad

    # Temperatura (piecewise constante, como ya lo tienes)
    T = float(T_grid[idx])

    # --- Cache: precomputar eventos una sola vez por simulación ---
    if not hasattr(zenteno_ode_variable, "_events_cache"):
        zenteno_ode_variable._events_cache = {}

    key = id(Nadd_grid)  # identifica este vector específico
    events = zenteno_ode_variable._events_cache.get(key)
    if events is None:
        events = extract_nadd_events(t_eval, Nadd_grid)
        zenteno_ode_variable._events_cache[key] = events

    # Nadd(t) suave con duración FIJA 1h, independiente del muestreo
    duration_h = 1.0
    k = 12.0
    Nadd = nadd_smooth_from_events(t, events, duration_h=duration_h, k=k)

    u = [T, Nadd]
    return zenteno_model_mod(t, x, u, params)
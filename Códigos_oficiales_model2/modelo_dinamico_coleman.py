"""
modelo_dinamico_coleman.py

Modelo dinámico de fermentación tipo Coleman/Cramer adaptado.

Estados: x = [X, N, S, E]

donde:
    X : biomasa activa efectiva [g/L]
    N : nitrógeno asimilable [g/L]
    S : azúcares totales [g/L]
    E : etanol [g/L]

Entradas: u = [T, Nadd]

donde:
    T    : temperatura [°C]
    Nadd : tasa de adición de nitrógeno [g/L/h]

Parámetros libres: p = [mu0, kd0, betaS0, Kn, Yxn, Yes, Ks]

donde:
    mu0    : factor multiplicativo de mu_max(T)
    kd0    : factor multiplicativo de k'_d(T)
    betaS0 : factor multiplicativo de betaSmax(T)
    Kn     : constante Monod de nitrógeno [g/L]
    Yxn    : rendimiento biomasa/nitrógeno [gX/gN]
    Yes    : rendimiento etanol/azúcar [gE/gS]
    Ks     : constante Michaelis-Menten de azúcar [g/L]

Notas:
    - Si mu0 = kd0 = betaS0 = 1, se recuperan las dependencias con temperatura reportadas por Coleman et al. (2007).
    - Se usa una única biomasa X, interpretada como biomasa activa efectiva.
"""

import numpy as np

# -------------------------------- Utilidades numéricas --------------------------------

EPS = 1e-9
BIG = 1e6


def safe_div(a, b, eps=EPS):
    return a / (b + eps)


def safe_exp(x, lo=-50.0, hi=50.0):
    """Exponencial con saturación para evitar overflow/underflow extremo."""
    return np.exp(np.clip(x, lo, hi))


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def _real_pos(z):
    """Toma parte real y fuerza no-negatividad."""
    r = float(np.real(z))
    return r if r > 0.0 else 0.0


# ------------------------- Suavizado de adición de nitrógeno -------------------------

def _sigmoid(z):
    """Sigmoide estable numéricamente."""
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def smooth_pulse_sigmoid(t, t0, duration_h, rate, k=12.0):
    """
    Pulso suave usando doble sigmoide.

    Aproxima un pulso rectangular:
        - comienza cerca de t0
        - termina cerca de t0 + duration_h
        - altura aproximada = rate [g/L/h]
    """
    t1 = t0 + duration_h
    return rate * (_sigmoid(k * (t - t0)) - _sigmoid(k * (t - t1)))


def extract_nadd_events(t_eval, Nadd_grid, eps=1e-12):
    """
    Extrae eventos de adición de nitrógeno desde Nadd_grid.

    Cada valor positivo en Nadd_grid se interpreta como una adición
    que comienza en el tiempo correspondiente de t_eval.
    """
    events = []

    for ti, ni in zip(t_eval, Nadd_grid):
        ni = float(ni)
        if ni > eps:
            events.append((float(ti), ni))

    return events


def nadd_smooth_from_events(t, events, duration_h=1.0, k=6.0):
    """
    Evalúa Nadd(t) como suma de pulsos suaves de duración fija.

    Por defecto, cada pulso dura 1 hora.
    """
    out = 0.0
    w = 4.0 / max(k, 1e-12)

    for t0, rate in events:
        # Evita evaluar pulsos que están muy lejos del tiempo actual
        if t < (t0 - 6 * w) or t > (t0 + duration_h + 6 * w):
            continue

        out += smooth_pulse_sigmoid(
            t=t,
            t0=t0,
            duration_h=duration_h,
            rate=rate,
            k=k
        )

    return float(out)


# ----------------------------- Modelo Coleman adaptado -----------------------------

def coleman_model(t, x, u, p, apply_nadd_in_model=True):
    """
    Modelo dinámico tipo Coleman/Cramer simplificado a 4 estados.

    Estados: x = [X, N, S, E]

    Entradas: u = [T_C, Nadd]

    Parámetros: p = [mu0, kd0, betaS0, Kn, Yxn, Yes, Ks]
    """

    # -------------------------
    # Entradas
    # -------------------------
    T_C = float(u[0])       # °C
    Nadd = float(u[1])      # g/L/h

    # -------------------------
    # Estados
    # -------------------------
    X = _real_pos(x[0])     # g/L, biomasa activa efectiva
    N = _real_pos(x[1])     # g/L, nitrógeno
    S = _real_pos(x[2])     # g/L, azúcar total
    E = _real_pos(x[3])     # g/L, etanol

    # -------------------------
    # Parámetros libres
    # -------------------------
    vals = [max(float(pi), EPS) for pi in p]

    mu0 = vals[0]
    kd0 = vals[1]
    betaS0 = vals[2]
    Kn = vals[3]
    Yxn = vals[4]
    Yes = vals[5]
    Ks = vals[6]

    # -------------------------
    # Seguridad temperatura
    # -------------------------
    T_C = float(clamp(T_C, 0.0, 60.0))

    # -------------------------
    # Parámetros dependientes de temperatura
    # -------------------------
    # Coleman et al. Tabla A.2:
    #
    # log(mu_max) = -3.92 + 0.0782*T
    # log(kd')    = -9.81 - 0.108*T + 0.00478*T^2
    # log(betaS)  = -2.98 + 0.0771*T
    #
    # Se agregan factores multiplicativos ajustables:
    # mu0, kd0, betaS0.

    mu_max = mu0 * safe_exp(-3.92 + 7.82e-2 * T_C)

    kd_prima = kd0 * safe_exp(-9.81 - 1.08e-1 * T_C + 4.78e-3 * T_C**2)

    betaS_max = betaS0 * safe_exp(-2.98 + 7.71e-2 * T_C)

    # -------------------------
    # Tasas específicas
    # -------------------------

    # Crecimiento limitado por nitrógeno
    mu = mu_max * safe_div(N, Kn + N)

    # Inactivación por etanol
    Kd = kd_prima * E

    # Consumo específico de azúcar
    betaS = betaS_max * safe_div(S, Ks + S)

    # -------------------------
    # EDOs
    # -------------------------

    # Biomasa activa efectiva
    dX = (mu - Kd) * X

    # Nitrógeno
    dN = -(mu * X) / Yxn

    if apply_nadd_in_model:
        dN += Nadd

    # Azúcar total
    dS = -(betaS * X) / Yes

    # Etanol
    dE = betaS * X

    # -------------------------
    # Seguridad numérica
    # -------------------------
    dX = float(clamp(dX, -BIG, BIG))
    dN = float(clamp(dN, -BIG, BIG))
    dS = float(clamp(dS, -BIG, BIG))
    dE = float(clamp(dE, -BIG, BIG))

    return np.array([dX, dN, dS, dE], dtype=float)


def coleman_ode_variable(t, x, params, t_eval, T_grid, Nadd_grid):
    """
    Wrapper compatible con solve_ivp.

    Usa:
        - T_grid como temperatura piecewise constante en °C.
        - Nadd_grid como eventos de adición de nitrógeno suavizados
          mediante pulsos sigmoidales.
    """

    # Índice del tramo actual
    idx = np.searchsorted(t_eval, t, side="right") - 1
    idx = int(np.clip(idx, 0, len(T_grid) - 1))

    # Temperatura en °C
    T_C = float(T_grid[idx])

    # -------------------------
    # Cache de eventos de Nadd
    # -------------------------
    if not hasattr(coleman_ode_variable, "_events_cache"):
        coleman_ode_variable._events_cache = {}

    key = id(Nadd_grid)

    events = coleman_ode_variable._events_cache.get(key)

    if events is None:
        events = extract_nadd_events(t_eval, Nadd_grid)
        coleman_ode_variable._events_cache[key] = events

    # Nadd(t) suavizado
    Nadd = nadd_smooth_from_events(t=t, events=events, duration_h=1.0, k=6.0)

    u = [T_C, Nadd]

    return coleman_model(t=t, x=x, u=u, p=params, apply_nadd_in_model=True)
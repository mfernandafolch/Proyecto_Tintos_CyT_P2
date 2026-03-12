"""
modelo_dinamico.py

Construcción del modelo dinámico (EDOs) de fermentación tipo Zenteno y su evaluación con entradas variables.

Incluye:
- Utilidades numéricas para estabilidad y robustez: safe_div, safe_exp, clamp, _real_pos.
- Sistema de EDOs del modelo Zenteno (estados x = [X, N, G, F, E]).
- Suavizado de la adición de nitrógeno Nadd mediante un pulso continuo (doble sigmoide) de duración fija
  (por defecto 1 hora), independiente del tiempo de muestreo de los datos.
- Función zenteno_ode_variable: wrapper compatible con solve_ivp que:
    (i) toma T desde una grilla temporal (piecewise constante),
    (ii) calcula Nadd(t) como función continua a partir de eventos (spikes) detectados en la grilla,
    (iii) llama a zenteno_model para retornar dx/dt.
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
def zenteno_model(t, x, u, p, apply_nadd_in_model=True):
    """
    Modelo de cinética de fermentación robusto.
    Estados: x = [X, N, G, F, E] (g/L)
    Entradas: u = [T (K), Nadd (g/L/h)]  (Nadd se suma a dN si apply_nadd_in_model=True)
    Parámetros: p[0..13] positivos.
    """
    # Entradas
    T = float(u[0])     # Kelvin
    Nadd = float(u[1])  # g/L/h     
            
    # Estados (no-negatividad y parte real)
    X = _real_pos(x[0])
    N = _real_pos(x[1])
    G = _real_pos(x[2])
    F = _real_pos(x[3])
    E = _real_pos(x[4])

    # Limitar T a un rango físico razonable (0-60 °C)
    T = clamp(T, 273.15, 333.15)

    # Parámetros (positivos)
    vals = [max(float(pi), EPS) for pi in p]
    (mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0,
     Yxn, Yxg, Yxf, Yeg, Yef) = vals

    # Constantes
    Cde     = 0.0415      # m3/kg E (Salmon 2003)
    Etd     = 130000.0    # kJ/kmol
    R       = 8.314       # kJ/kmol/K (Boulton 1979)
    Eac     = 59453.0     # kJ/kmol (Boulton 1979)
    Eafe    = 11000.0     # kJ/kmol (Zenteno 2010)
    EaKn    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKg    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKf    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKig   = 46055.0     # kJ/kmol (Boulton 1979)
    EaKie   = 46055.0     # kJ/kmol (Boulton 1979)
    Eam     = 37681.0     # kJ/kmol (Boulton 1979)
    m0      = 0.01        # kgS/kg bio/h

    # Arrhenius con exponente acotado
    mu_max    = mu0   * safe_exp(Eac *(T-300.00)/(300.00*R*T))
    betaG_max = betaG0* safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    betaF_max = betaF0* safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    Kn        = Kn0   * safe_exp(EaKn*(T-293.15)/(293.15*R*T))
    Kg        = Kg0   * safe_exp(EaKg*(T-293.15)/(293.15*R*T))
    Kf        = Kf0   * safe_exp(EaKf*(T-293.15)/(293.15*R*T))
    Kig       = Kig0  * safe_exp(EaKig*(T-293.15)/(293.15*R*T))
    Kie       = Kie0  * safe_exp(EaKie*(T-293.15)/(293.15*R*T))
    m         = m0    * safe_exp(Eam *(T-293.30)/(293.30*R*T))

    # Tasas con divisiones seguras
    mu      = mu_max * safe_div(N, N + Kn)
    beta_G  = betaG_max* safe_div(G, G + Kg) * safe_div(Kie, E + Kie)
    beta_F  = betaF_max* safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)

    # Temperatura de muerte térmica con E acotado (evita E**3 enorme)
    E_cap = clamp(E, 0.0, 200.0)
    Td = -0.0001*(E_cap**3) + 0.0049*(E_cap**2) - 0.1279*E_cap + 315.89
    Td = clamp(Td, 273.15, 333.15)

    # Tasa de muerte específica con exponencial segura
    if T >= Td:
        exponent = (Cde*E_cap) + safe_div(Etd*(T-305.65), (305.65*R*T))
        Kd = Kd0 * safe_exp(exponent, lo=-50.0, hi=50.0)
    else:
        Kd = 0.0

    # Mezcla para mantenimiento (evita 0/0)
    GpF = G + F + EPS
    mG = G / GpF
    mF = F / GpF

    # EDOs
    dX = (mu - Kd) * X
    dN = -(mu / max(Yxn, EPS)) * X
    if apply_nadd_in_model:
        dN += Nadd  # Nadd se agrega como tasa en el paso correspondiente
    dG = -((mu / max(Yxg, EPS)) + (beta_G / max(Yeg, EPS)) + m*mG) * X
    dF = -((mu / max(Yxf, EPS)) + (beta_F / max(Yef, EPS)) + m*mF) * X
    dE = (beta_G + beta_F) * X

    dX = float(clamp(dX, -BIG, BIG))
    dN = float(clamp(dN, -BIG, BIG))
    dG = float(clamp(dG, -BIG, BIG))
    dF = float(clamp(dF, -BIG, BIG))
    dE = float(clamp(dE, -BIG, BIG))
    return np.array([dX, dN, dG, dF, dE], dtype=float)


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
    return zenteno_model(t, x, u, params)
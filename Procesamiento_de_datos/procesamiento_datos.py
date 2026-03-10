"""
procesamiento_datos.py

Funciones para procesar los datos extraídos del Excel.

Incluye:
- Cálculo de condiciones iniciales (X, N, G, F, E en g/L) y etanol final observado
- Suavizado con splines usando ruido estimado vía Savitzky-Golay (como en el notebook)
- Cálculo de t_start_opt (max entre primer dato de densidad y subida de setpoint 27–29°C)
- Remuestreo cada t_muestreo_h (e.g., 8 h) y evaluación de perfiles suavizados
- Construcción de vector Nadd (g/L): ceros salvo impulso (sesgado hacia adelante) en 2ª adición de FDA,
  asegurando que ocurra DESPUÉS del cruce de densidad (<= densidad_objetivo o 1070 por defecto)
- Entrega de dos vectores de tiempo: t_abs (absoluto) y t_rel (ajustado, inicia en 0)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from extraccion_datos import FermentationData, load_fermentation_data


# =========================
# Dataclasses de salida
# =========================
@dataclass
class InitialConditions:
    X0_gL: float
    N0_gL: float
    G0_gL: float
    F0_gL: float
    E0_gL: float
    E_final_obs_gL: float


@dataclass
class ProcessedProfiles:
    # tiempo absoluto (horas desde el origen del Excel)
    t_abs_h: np.ndarray
    # tiempo ajustado (t=0 en t_start_opt)
    t_rel_h: np.ndarray

    # perfiles procesados
    densidad: np.ndarray
    azucar: np.ndarray
    temp_mosto: np.ndarray
    temp_sombrero: np.ndarray
    temp_promedio: np.ndarray
    setpoint: np.ndarray

    # adición de N (g/L) como impulso discreto
    Nadd_gL: np.ndarray

    # info extra
    t_inicio_opt_h: float


@dataclass
class ProcessedFermentation:
    init: InitialConditions
    profiles: ProcessedProfiles
    meta: Dict[str, float]


# =========================
# Helpers: conversiones
# =========================
def densidad_to_sugar(densidad):
    """Convierte densidad a concentración de azúcar (g/L)."""
    return 2.5616 * np.asarray(densidad, dtype=float) - 2577.4


def brix_to_total_sugar_gL(brix: float) -> float:
    """Convierte °Brix a azúcar total (g/L)."""
    if brix is None or np.isnan(brix):
        return np.nan
    return 12.0 * brix - 40.0


# =========================
# Helpers: condiciones iniciales
# =========================
def calc_vol_total(
    vol_mosto_est_L: float,
    vol_sang_L: float,
    vol_tart_L: float,
    vol_freek_L: float,
    vol_agua_L: float,
    vol_conc_L: float,
    vol_fda_L: float
) -> Tuple[float, float]:
    vol_post = vol_mosto_est_L - vol_sang_L
    vol_total = vol_post + vol_tart_L + vol_freek_L + vol_agua_L + vol_conc_L + vol_fda_L
    return vol_post, vol_total


def calc_conc_yan_inicial_mgL(
    yan0_mgL: float,
    vol_post_sangria_L: float,
    vol_total_corr_L: float,
    dosis_fda_g_hL: float,
    fda_factor: float = 25/10
) -> float:
    """
    Ajuste del YAN inicial considerando:
    - YAN se reescala por cambio de volumen (post sangría)
    - se agrega aporte FDA convertido a mg/L de YAN equivalente (dosis * 25/10)
    - se normaliza por vol_total_corr
    """
    if np.isnan(yan0_mgL) or vol_total_corr_L <= 0 or vol_post_sangria_L <= 0:
        return np.nan

    fda_added_mgL = dosis_fda_g_hL * fda_factor
    ajuste_yan_vol = vol_post_sangria_L * yan0_mgL
    conc = (ajuste_yan_vol + fda_added_mgL * vol_total_corr_L) / vol_total_corr_L
    return conc


def calc_biomasa_inicial_gL(
    vol_total_corr_L: float,
    vol_levadura_L: float,
    poblacion_cel_mL: float,
    g_per_cell: float = 2.8571e-11,
    default_X: float = 0.0595
) -> float:
    if vol_total_corr_L <= 0:
        return default_X
    if (poblacion_cel_mL is None) or np.isnan(poblacion_cel_mL) or poblacion_cel_mL <= 0:
        return default_X

    conc_inoc_gL = poblacion_cel_mL * g_per_cell * 1000.0
    masa_total_g = conc_inoc_gL * vol_levadura_L
    X0 = masa_total_g / vol_total_corr_L
    return X0 if (not np.isnan(X0) and X0 > 0) else default_X


# =========================
# Helpers: ruido (Savitzky-Golay) y splines
# =========================
def estimate_noise_residual(y, window: int, poly: int) -> float:
    """
    Ruido efectivo via Savitzky-Golay fuerte + sigma de residuos.
    Funciona con Series o ndarray.
    """
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return np.nan

    y = np.asarray(pd.to_numeric(y, errors="coerce"), dtype=float)
    mask = ~np.isnan(y)
    yv = y[mask]

    if len(yv) < window:
        return np.nan

    w = int(window)
    if w % 2 == 0:
        w += 1
    if w <= poly:
        w = poly + 3
        if w % 2 == 0:
            w += 1

    y_smooth = savgol_filter(yv, window_length=w, polyorder=int(poly))
    return float(np.std(yv - y_smooth, ddof=1))


def _spline_s_from_sigma(sigma: float, N: int, alpha: float) -> float:
    """s = alpha * N * sigma^2 (notebook)."""
    if (sigma is None) or np.isnan(sigma) or N <= 0:
        return 0.0
    return float(max(alpha * N * (sigma ** 2), 0.0))


def fit_spline(t_h: np.ndarray, y: np.ndarray, s: float, k: int = 2):
    """Ajusta UnivariateSpline en puntos no-NaN."""
    from scipy.interpolate import UnivariateSpline
    t_h = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(t_h) & ~np.isnan(y)
    if mask.sum() < (k + 2):
        return None
    return UnivariateSpline(t_h[mask], y[mask], s=float(s), k=int(k))


def _interp_fill_1d(y: np.ndarray) -> np.ndarray:
    """Rellena NaNs por interpolación lineal (como notebook para setpoint)."""
    s = pd.Series(pd.to_numeric(y, errors="coerce"))
    return s.interpolate(limit_direction="both").to_numpy(dtype=float)


def compute_t_start_opt(t_h: np.ndarray, dens: np.ndarray, setpoint: np.ndarray) -> float:
    """
    Replica tu lógica:
    - first_density_hour: primer tiempo con densidad no-NaN
    - sp_start_hour: tiempo donde setpoint entra al rango 27–29 y antes estaba <27 (primer "up")
    - t_start_opt = max(first_density_hour, sp_start_hour)
    """
    t_h = np.asarray(t_h, dtype=float)
    dens = np.asarray(dens, dtype=float)
    sp = np.asarray(setpoint, dtype=float)

    idx_first_density = np.where(~np.isnan(dens))[0]
    if len(idx_first_density) == 0:
        raise ValueError("No hay mediciones de densidad válidas en 'Prov Sensores'.")
    first_density_hour = float(t_h[idx_first_density[0]])

    sp_clean = _interp_fill_1d(sp)

    idx_sp_range = np.where((sp_clean >= 27.0) & (sp_clean <= 29.0))[0]
    idx_sp_up = None
    for idx in idx_sp_range:
        prev = sp_clean[idx - 1] if idx > 0 else np.nan
        if np.isnan(prev) or prev < 27.0:
            idx_sp_up = int(idx)
            break
    if idx_sp_up is None and len(idx_sp_range) > 0:
        idx_sp_up = int(idx_sp_range[0])

    sp_start_hour = float(t_h[idx_sp_up]) if idx_sp_up is not None else first_density_hour
    return float(max(first_density_hour, sp_start_hour))


def build_t_opt(t_start_opt: float, t_end: float, t_muestreo_h: float) -> np.ndarray:
    """t_opt = arange(t_start_opt, t_end, t_muestreo) + asegura incluir t_end."""
    if t_muestreo_h <= 0:
        raise ValueError("t_muestreo_h debe ser > 0.")
    t_opt = np.arange(float(t_start_opt), float(t_end) + 1e-6, float(t_muestreo_h))
    if len(t_opt) == 0:
        t_opt = np.array([float(t_start_opt)], dtype=float)
    if t_opt[-1] < float(t_end):
        t_opt = np.append(t_opt, float(t_end))
    return t_opt


def find_first_time_below_threshold(
    spline_dens,
    t_start: float,
    t_end: float,
    threshold: float,
    dt_fine_h: float = 0.25
) -> float:
    """
    Busca el primer tiempo (abs, h) en [t_start, t_end] donde densidad_suavizada <= threshold.
    Retorna np.nan si no se encuentra cruce o si spline_dens es None.
    """
    if spline_dens is None:
        return np.nan
    if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
        return np.nan
    if np.isnan(threshold):
        return np.nan

    t_fine = np.arange(float(t_start), float(t_end) + 1e-9, float(dt_fine_h))
    if len(t_fine) < 2:
        return np.nan

    dens_fine = spline_dens(t_fine)
    mask = ~np.isnan(dens_fine)
    if mask.sum() < 2:
        return np.nan

    t_fine = t_fine[mask]
    dens_fine = dens_fine[mask]

    idx = np.where(dens_fine <= threshold)[0]
    if len(idx) == 0:
        return np.nan

    return float(t_fine[int(idx[0])])


# =========================
# Nadd (impulso por 2ª FDA) - snap hacia adelante
# =========================
def build_Nadd_impulse_forward(
    t_opt_abs: np.ndarray,
    t_event_abs: float,
    yan_fda2_mgL: float,
    t_cross_abs: float = np.nan,
    strict_after_cross: bool = True
) -> Tuple[np.ndarray, float, int]:
    """
    Nadd (g/L) como impulso discreto en t_opt_abs:

    - todo 0
    - ubica el impulso en el PRIMER t_opt_abs que sea >= t_event_abs (sesgo hacia adelante)
    - además asegura estar DESPUÉS del cruce (si t_cross_abs existe):
        strict_after_cross=True  -> t_discreto >  t_cross_abs
        strict_after_cross=False -> t_discreto >= t_cross_abs

    Retorna:
      (Nadd_vector, t_event_discreto_abs, idx_event)
    """
    t_opt_abs = np.asarray(t_opt_abs, dtype=float)
    Nadd = np.zeros_like(t_opt_abs, dtype=float)

    if (yan_fda2_mgL is None) or np.isnan(yan_fda2_mgL) or yan_fda2_mgL <= 0:
        return Nadd, np.nan, -1
    if (t_event_abs is None) or np.isnan(t_event_abs):
        return Nadd, np.nan, -1
    if t_opt_abs.size == 0 or np.all(np.isnan(t_opt_abs)):
        return Nadd, np.nan, -1

    t_min = float(t_event_abs)
    if (t_cross_abs is not None) and (not np.isnan(t_cross_abs)):
        if strict_after_cross:
            t_min = max(t_min, float(t_cross_abs) + 1e-9)
        else:
            t_min = max(t_min, float(t_cross_abs))

    idx_candidates = np.where(t_opt_abs >= t_min)[0]
    if len(idx_candidates) == 0:
        return Nadd, np.nan, -1

    idx = int(idx_candidates[0])
    Nadd[idx] = float(yan_fda2_mgL) / 1000.0
    return Nadd, float(t_opt_abs[idx]), idx


# =========================
# PIPELINE MAESTRO
# =========================
def process_excel(
    path_excel: str,
    t_muestreo_h: float = 8.0,
    noise_cfg: Optional[Dict[str, Dict[str, int]]] = None,
    spline_cfg: Optional[Dict[str, float]] = None,
    t_end_h: Optional[float] = None,
) -> ProcessedFermentation:

    if noise_cfg is None:
        noise_cfg = {"temp": {"window": 81, "poly": 1}, "dens": {"window": 51, "poly": 1}}
    if spline_cfg is None:
        spline_cfg = {"alpha_temp": 1.0, "alpha_dens": 1.0}

    raw: FermentationData = load_fermentation_data(path_excel)

    # -------------------------
    # Condiciones iniciales
    # -------------------------
    vol_post, vol_total = calc_vol_total(
        raw.antecedentes.vol_mosto_est_L,
        raw.insumos.vol_sang_L,
        raw.insumos.vol_tart_L,
        raw.insumos.vol_freek_L,
        raw.insumos.vol_agua_L,
        raw.insumos.vol_conc_L,
        raw.insumos.fda.vol_FDA_L
    )

    Az_total_gL = brix_to_total_sugar_gL(raw.antecedentes.brix_inicial)
    G0 = Az_total_gL / 2.0 if not np.isnan(Az_total_gL) else np.nan
    F0 = Az_total_gL / 2.0 if not np.isnan(Az_total_gL) else np.nan

    yan_ini_mgL = calc_conc_yan_inicial_mgL(
        yan0_mgL=raw.laboratorio.yan0_mgL,
        vol_post_sangria_L=vol_post,
        vol_total_corr_L=vol_total,
        dosis_fda_g_hL=raw.insumos.fda.dosis_FDA_g_hL
    )
    N0 = yan_ini_mgL / 1000.0 if not np.isnan(yan_ini_mgL) else np.nan

    X0 = calc_biomasa_inicial_gL(
        vol_total_corr_L=vol_total,
        vol_levadura_L=raw.insumos.vol_levadura_L,
        poblacion_cel_mL=raw.insumos.poblacion_levadura_cel_mL
    )

    init = InitialConditions(
        X0_gL=float(X0),
        N0_gL=float(N0) if not np.isnan(N0) else np.nan,
        G0_gL=float(G0) if not np.isnan(G0) else np.nan,
        F0_gL=float(F0) if not np.isnan(F0) else np.nan,
        E0_gL=0.0,
        E_final_obs_gL=float(raw.laboratorio.E_final_obs_gL) if not np.isnan(raw.laboratorio.E_final_obs_gL) else np.nan
    )

    # -------------------------
    # Series crudas
    # -------------------------
    t_h = np.asarray(raw.sensores.t_h, dtype=float)
    dens = np.asarray(raw.sensores.densidad, dtype=float)
    tm = np.asarray(raw.sensores.temp_mosto, dtype=float)
    ts = np.asarray(raw.sensores.temp_sombrero, dtype=float)
    tp = np.asarray(raw.sensores.temp_promedio_raw, dtype=float)
    sp = np.asarray(raw.sensores.temp_setpoint, dtype=float)

    t_end = float(np.nanmax(t_h)) if t_end_h is None else float(t_end_h)

    # -------------------------
    # Ruido y s
    # -------------------------
    sigma_temp = estimate_noise_residual(tm, noise_cfg["temp"]["window"], noise_cfg["temp"]["poly"])
    sigma_dens = estimate_noise_residual(dens, noise_cfg["dens"]["window"], noise_cfg["dens"]["poly"])

    N_temp = int(np.sum(~np.isnan(tm)))
    N_dens = int(np.sum(~np.isnan(dens)))

    s_temp = _spline_s_from_sigma(sigma_temp, N_temp, float(spline_cfg.get("alpha_temp", 1.0)))
    s_dens = _spline_s_from_sigma(sigma_dens, N_dens, float(spline_cfg.get("alpha_dens", 1.0)))

    # -------------------------
    # Splines
    # -------------------------
    try:
        spline_dens = fit_spline(t_h, dens, s_dens, k=2)
        spline_tm = fit_spline(t_h, tm, s_temp, k=2)
        spline_ts = fit_spline(t_h, ts, s_temp, k=2)
        spline_tp = fit_spline(t_h, tp, s_temp, k=2)
    except Exception:
        spline_dens = spline_tm = spline_ts = spline_tp = None

    # -------------------------
    # Inicio optimizado + grilla
    # -------------------------
    t_start_opt = compute_t_start_opt(t_h, dens, sp)

    t_opt_abs = build_t_opt(t_start_opt, t_end, t_muestreo_h)
    t_opt_rel = t_opt_abs - float(t_start_opt)

    # -------------------------
    # FDA2 / Nadd: SIEMPRE evento = cruce + 3h (fecha solo diagnóstico)
    # -------------------------
    yan_fda2_mgL = float(getattr(raw.insumos.fda, "yan_FDA_2_mgL", 0.0))
    horas_post_fda2 = float(getattr(raw.insumos.fda, "horas_post_FDA_2_h", 0.0))

    # diagnóstico: "tiempo según Excel" (si existiera)
    t_fecha_abs = np.nan
    if (not np.isnan(horas_post_fda2)) and (horas_post_fda2 > 0):
        t_fecha_abs = float(t_start_opt) + float(horas_post_fda2)

    dens_obj = getattr(raw.insumos.fda, "densidad_objetivo_FDA_2", np.nan)
    threshold = float(dens_obj) if (dens_obj is not None and not np.isnan(dens_obj) and dens_obj > 0) else 1070.0

    t_cross_abs = np.nan
    t_event_abs = np.nan  # continuo (cruce+3)
    t_cross_plus_abs = np.nan

    if yan_fda2_mgL > 0:
        t_cross_abs = find_first_time_below_threshold(
            spline_dens=spline_dens,
            t_start=float(t_start_opt),
            t_end=float(t_end),
            threshold=float(threshold),
            dt_fine_h=0.25
        )

        if not np.isnan(t_cross_abs):
            t_cross_plus_abs = float(t_cross_abs) + 3.0
            t_event_abs = float(t_cross_plus_abs)

            # print("[FDA2] REGLA: evento = cruce + 3h (SIEMPRE)")
            # print(f"[FDA2] threshold={threshold:.2f} | t_cruce_abs={t_cross_abs:.2f} h | t_evento_abs={t_event_abs:.2f} h")
            # if not np.isnan(t_fecha_abs):
                # print(f"[FDA2] (diagnóstico) t_fecha_excel_abs={t_fecha_abs:.2f} h | delta_excel_vs_evento={t_fecha_abs - t_event_abs:.2f} h")
            # print("")
        # else:
            # print("[FDA2] No se encontró cruce de densidad bajo el umbral -> no se puede ubicar FDA2/Nadd.\n")

    # SNAP ADELANTE (y estrictamente después del cruce)
    Nadd_opt, t_event_discreto_abs, idx_event_used = build_Nadd_impulse_forward(
        t_opt_abs=t_opt_abs,
        t_event_abs=t_event_abs,
        yan_fda2_mgL=yan_fda2_mgL,
        t_cross_abs=t_cross_abs,
        strict_after_cross=True
    )

    #if (idx_event_used >= 0) and (not np.isnan(t_event_discreto_abs)):
        #print(f"[FDA2] Evento discreto en grilla: t={t_event_discreto_abs:.2f} h (rel={t_event_discreto_abs - t_start_opt:.2f} h) | Nadd={np.max(Nadd_opt):.6f} g/L\n")

    # -------------------------
    # Evaluación perfiles en t_opt_abs
    # -------------------------
    if spline_dens is not None:
        dens_opt = spline_dens(t_opt_abs)
    else:
        mask_d = ~np.isnan(dens)
        dens_opt = np.interp(t_opt_abs, t_h[mask_d], dens[mask_d]) if mask_d.sum() >= 2 else np.full_like(t_opt_abs, np.nan)

    def _eval_spline_or_interp(y_raw, spline_obj):
        if spline_obj is not None:
            return spline_obj(t_opt_abs)
        mask = ~np.isnan(y_raw)
        return np.interp(t_opt_abs, t_h[mask], y_raw[mask]) if mask.sum() >= 2 else np.full_like(t_opt_abs, np.nan, dtype=float)

    tm_opt = _eval_spline_or_interp(tm, spline_tm)
    ts_opt = _eval_spline_or_interp(ts, spline_ts)
    tp_opt = _eval_spline_or_interp(tp, spline_tp)

    azucar_opt = densidad_to_sugar(dens_opt)

    sp_clean = _interp_fill_1d(sp)
    setpoint_opt = np.interp(t_opt_abs, t_h, sp_clean)

    profiles = ProcessedProfiles(
        t_abs_h=np.asarray(t_opt_abs, dtype=float),
        t_rel_h=np.asarray(t_opt_rel, dtype=float),

        densidad=np.asarray(dens_opt, dtype=float),
        azucar=np.asarray(azucar_opt, dtype=float),
        temp_mosto=np.asarray(tm_opt, dtype=float),
        temp_sombrero=np.asarray(ts_opt, dtype=float),
        temp_promedio=np.asarray(tp_opt, dtype=float),
        setpoint=np.asarray(setpoint_opt, dtype=float),

        Nadd_gL=np.asarray(Nadd_opt, dtype=float),

        t_inicio_opt_h=float(t_start_opt),
    )

    # -------------------------
    # META (mantengo lo otro + dejo solo lo útil de Nadd/FDA2)
    # -------------------------
    meta = {
        "vol_post_sangria_L": float(vol_post),
        "vol_total_corr_L": float(vol_total),
        "yan_ini_mgL": float(yan_ini_mgL) if not np.isnan(yan_ini_mgL) else np.nan,
        "sigma_temp": float(sigma_temp) if not np.isnan(sigma_temp) else np.nan,
        "sigma_dens": float(sigma_dens) if not np.isnan(sigma_dens) else np.nan,
        "s_temp": float(s_temp),
        "s_dens": float(s_dens),
        "t_muestreo_h": float(t_muestreo_h),
        "t_start_opt_abs_h": float(t_start_opt),

        # --- Nadd útil (discreto en grilla) ---
        "Nadd_event_index": float(idx_event_used),
        "Nadd_event_time_abs_h": float(t_event_discreto_abs) if not np.isnan(t_event_discreto_abs) else np.nan,
        # ESTE ES EL QUE IMPORTA:
        "Nadd_event_time_rel_h": float(t_event_discreto_abs - t_start_opt) if not np.isnan(t_event_discreto_abs) else np.nan,
        "Nadd_value_gL": float(np.max(Nadd_opt)) if idx_event_used >= 0 else 0.0,

        # --- decisión FDA2 (mínimo) ---
        "FDA2_threshold_densidad": float(threshold),
        "FDA2_t_cruce_abs_h": float(t_cross_abs) if not np.isnan(t_cross_abs) else np.nan,
        "FDA2_t_evento_corregido_abs_h": float(t_event_abs) if not np.isnan(t_event_abs) else np.nan,  # cruce+3h (continuo)
        "FDA2_t_fecha_abs_h": float(t_fecha_abs) if not np.isnan(t_fecha_abs) else np.nan,              # solo diagnóstico
    }

    return ProcessedFermentation(init=init, profiles=profiles, meta=meta)
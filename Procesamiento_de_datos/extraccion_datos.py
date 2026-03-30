"""
extraccion_datos.py

Funciones para la extracción de datos desde los archivos Excel de fermentaciones industriales de vino.

Incluye:
- Data puntual: Antecedentes, Laboratorio, Insumos Operacionales (+ lógica FDA compleja y Otros insumos)
- Series temporales: Prov Sensores (tiempo en horas, densidad, temperaturas)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =========================
# CLASES (DATACLASSES)
# =========================
@dataclass
class AntecedentesData:
    vol_mosto_est_L: float
    brix_inicial: float


@dataclass
class LaboratorioData:
    yan0_mgL: float
    alcohol_grado: float
    E_final_obs_gL: float


@dataclass
class FDAData:
    # Primera adición
    vol_FDA_L: float
    dosis_FDA_g_hL: float
    fecha_FDA: pd.Timestamp

    # Segunda adición (si existe)
    vol_FDA_2_L: float
    dosis_FDA_2_g_hL: float
    fecha_FDA_2: pd.Timestamp
    horas_post_FDA_2_h: float

    # Densidad objetivo (gatillante) para la 2ª adición
    densidad_objetivo_FDA_2: float  # np.nan si no existe

    @property
    def yan_FDA_mgL(self) -> float:
        return self.dosis_FDA_g_hL * 25 / 10

    @property
    def yan_FDA_2_mgL(self) -> float:
        return self.dosis_FDA_2_g_hL * 25 / 10


@dataclass
class InsumosData:
    vol_sang_L: float
    vol_freek_L: float
    vol_tart_L: float
    vol_agua_L: float
    vol_conc_L: float

    vol_levadura_L: float
    poblacion_levadura_cel_mL: float
    tipo_inoculo: str   # NUEVO

    fda: FDAData


@dataclass
class ProvSensoresData:
    t_h: np.ndarray
    densidad: np.ndarray
    temp_mosto: np.ndarray
    temp_sombrero: np.ndarray
    temp_setpoint: np.ndarray
    temp_promedio_raw: np.ndarray


@dataclass
class FermentationData:
    antecedentes: AntecedentesData
    laboratorio: LaboratorioData
    insumos: InsumosData
    sensores: ProvSensoresData


# =========================
# HELPERS
# =========================
def to_float(x):
    """Convierte strings con coma decimal y números; si falla -> np.nan."""
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return np.nan


def find_first_match(df: pd.DataFrame, keyword: str):
    """Retorna (row, col) de la primera aparición (case-insensitive) de keyword en df."""
    mask = df.astype(str).apply(lambda col: col.str.contains(keyword, case=False, na=False))
    rows, cols = np.where(mask.values)
    if len(rows) == 0:
        raise ValueError(f"No se encontró '{keyword}' en la hoja.")
    return int(rows[0]), int(cols[0])


def find_all_matches(df: pd.DataFrame, keyword: str):
    """Retorna lista de (row, col) donde aparece keyword (case-insensitive)."""
    mask = df.astype(str).apply(lambda col: col.str.contains(keyword, case=False, na=False))
    rows, cols = np.where(mask.values)
    return list(zip(rows.tolist(), cols.tolist()))


def value_at_offset(df: pd.DataFrame, keyword: str, col_offset: int, row_offset: int = 0, default=np.nan):
    """Busca keyword y devuelve el valor en (row+row_offset, col+col_offset) convertido a float."""
    r, c = find_first_match(df, keyword)
    rr, cc = r + row_offset, c + col_offset
    if rr < 0 or cc < 0 or rr >= df.shape[0] or cc >= df.shape[1]:
        return default
    return to_float(df.iat[rr, cc])


def fix_fda_dose(dosis: float, threshold: float = 100.0) -> float:
    """Corrige dosis FDA si viene 10x inflada (>threshold => /10)."""
    if pd.isna(dosis) or dosis <= 0:
        return 0.0
    return dosis / 10.0 if dosis > threshold else dosis


def is_unit_L(unit_raw) -> bool:
    unit = str(unit_raw).strip().lower().replace(" ", "")
    return unit in ["l", "(l)", "litro", "litros"]


def is_unit_kg(unit_raw) -> bool:
    unit = str(unit_raw).strip().lower().replace(" ", "")
    return unit in ["kg", "(kg)", "kilogramo", "kilogramos"]


def compute_hours_diff_with_window(fecha_1, fecha_2, extra_window_h: float = 12.0) -> float:
    """Horas entre fechas (no negativo) + ventana fija (12h)."""
    if pd.isna(fecha_1) or pd.isna(fecha_2):
        return 0.0
    delta_h = (fecha_2 - fecha_1).total_seconds() / 3600.0
    return max(delta_h, 0.0) + extra_window_h


# =========================
# EXTRACCIÓN: ANTECEDENTES / LABORATORIO
# =========================
def extract_antecedentes(path_excel: str) -> AntecedentesData:
    df_ant = pd.read_excel(path_excel, sheet_name="Antecedentes")
    vol_mosto = to_float(df_ant.loc[0, "ant_vino_estimado_l"])
    brix_ini = to_float(df_ant.loc[0, "oper_inicio_brix"])
    return AntecedentesData(vol_mosto_est_L=vol_mosto, brix_inicial=brix_ini)


def extract_laboratorio(path_excel: str) -> LaboratorioData:
    df_lab = pd.read_excel(path_excel, sheet_name="Laboratorio")
    yan0 = value_at_offset(df_lab, "YAN", col_offset=4, default=np.nan)      # mg/L
    alc = value_at_offset(df_lab, "Alcohol", col_offset=4, default=np.nan)  # ° o %

    E_final = (alc / 100.0) * 789.3 if not np.isnan(alc) else np.nan
    return LaboratorioData(yan0_mgL=yan0, alcohol_grado=alc, E_final_obs_gL=E_final)


# =========================
# FDA COMPLETO (según la lógica)
# =========================
def _read_fda_primary_from_insumos(df_ins: pd.DataFrame):
    matches = sorted(find_all_matches(df_ins, "FDA"), key=lambda rc: rc[0])
    if not matches:
        return None

    r, c = matches[0]

    unit_raw = df_ins.iat[r, c + 2] if (c + 2) < df_ins.shape[1] else ""
    vol_L = 0.0
    if is_unit_L(unit_raw):
        vol_raw = df_ins.iat[r, c + 1] if (c + 1) < df_ins.shape[1] else np.nan
        vol_val = to_float(vol_raw)
        vol_L = 0.0 if (pd.isna(vol_val) or vol_val <= 0) else vol_val
    elif is_unit_kg(unit_raw):
        vol_L = 0.0
    else:
        return None

    dosis_raw = df_ins.iat[r, c + 4] if (c + 4) < df_ins.shape[1] else np.nan
    dosis = fix_fda_dose(to_float(dosis_raw), threshold=100.0)

    fecha_raw = df_ins.iat[r, c + 7] if (c + 7) < df_ins.shape[1] else None
    fecha = pd.to_datetime(fecha_raw, errors="coerce", dayfirst=True)

    return {"row": r, "col": c, "unit": unit_raw, "vol_L": vol_L, "dosis_g_hL": dosis, "fecha": fecha}


def _read_fda_second_from_insumos(df_ins: pd.DataFrame):
    matches = sorted(find_all_matches(df_ins, "FDA"), key=lambda rc: rc[0])
    if len(matches) < 2:
        return None

    for (r, c) in matches[1:]:
        durante_raw = df_ins.iat[r, c + 12] if (c + 12) < df_ins.shape[1] else ""
        durante_txt = str(durante_raw).strip().lower()
        if "durante" not in durante_txt:
            continue

        unit_raw = df_ins.iat[r, c + 2] if (c + 2) < df_ins.shape[1] else ""
        vol_L = 0.0
        if is_unit_L(unit_raw):
            vol_raw = df_ins.iat[r, c + 1] if (c + 1) < df_ins.shape[1] else np.nan
            vol_val = to_float(vol_raw)
            vol_L = 0.0 if (pd.isna(vol_val) or vol_val <= 0) else vol_val

        dosis = 0.0
        if (c + 4) < df_ins.shape[1]:
            dosis = fix_fda_dose(to_float(df_ins.iat[r, c + 4]), threshold=100.0)

        if dosis <= 0:
            dosis = 19.5

        fecha_raw = df_ins.iat[r, c + 7] if (c + 7) < df_ins.shape[1] else None
        fecha = pd.to_datetime(fecha_raw, errors="coerce", dayfirst=True)

        # Densidad objetivo en columna +8 (c/r a FDA) para el caso Insumos Operacionales
        dens_obj_raw = df_ins.iat[r, c + 8] if (c + 8) < df_ins.shape[1] else np.nan
        dens_obj = to_float(dens_obj_raw)
        if pd.isna(dens_obj) or dens_obj <= 0:
            dens_obj = np.nan

        return {
            "row": r, "col": c, "unit": unit_raw,
            "vol_L": vol_L, "dosis_g_hL": dosis, "fecha": fecha,
            "densidad_objetivo": dens_obj,
            "source": "insumos"
        }

    return None


def _read_fda_second_from_otros(df_otro: pd.DataFrame):
    matches = sorted(find_all_matches(df_otro, "FDA"), key=lambda rc: rc[0])
    if not matches:
        return None

    for (r, c) in matches:
        # densidad objetivo (gatillante) en +3
        dens_raw = df_otro.iat[r, c + 3] if (c + 3) < df_otro.shape[1] else np.nan
        dens_val = to_float(dens_raw)
        if pd.isna(dens_val) or dens_val <= 0:
            continue  # no es secundaria según tu regla

        unit_raw = df_otro.iat[r, c + 2] if (c + 2) < df_otro.shape[1] else ""
        vol_L = 0.0
        if is_unit_L(unit_raw):
            vol_raw = df_otro.iat[r, c + 1] if (c + 1) < df_otro.shape[1] else np.nan
            vol_val = to_float(vol_raw)
            vol_L = 0.0 if (pd.isna(vol_val) or vol_val <= 0) else vol_val

        fecha_raw = df_otro.iat[r, c + 5] if (c + 5) < df_otro.shape[1] else None
        fecha = pd.to_datetime(fecha_raw, errors="coerce", dayfirst=True)
        
        # Dosis se asigna automáticamente porque no trae este dato la planilla
        dosis = 19.5 
    
        return {
            "row": r, "col": c, "unit": unit_raw,
            "vol_L": vol_L, "dosis_g_hL": dosis, "fecha": fecha,
            "densidad_objetivo": dens_val,
            "source": "otros"
        }

    return None


def extract_fda_complex_from_excel(path_excel: str) -> FDAData:
    df_ins = pd.read_excel(path_excel, sheet_name="Insumos Operacionales")
    try:
        df_otro = pd.read_excel(path_excel, sheet_name="Otros insumos")
    except Exception:
        df_otro = None

    primary = _read_fda_primary_from_insumos(df_ins)

    if primary is None:
        return FDAData(
            vol_FDA_L=0.0, dosis_FDA_g_hL=0.0, fecha_FDA=pd.NaT,
            vol_FDA_2_L=0.0, dosis_FDA_2_g_hL=0.0, fecha_FDA_2=pd.NaT,
            horas_post_FDA_2_h=0.0,
            densidad_objetivo_FDA_2=np.nan
        )

    second = _read_fda_second_from_insumos(df_ins)
    if second is None and df_otro is not None:
        second = _read_fda_second_from_otros(df_otro)

    vol_1 = primary["vol_L"]
    dosis_1 = primary["dosis_g_hL"]
    fecha_1 = primary["fecha"]

    if second is None:
        return FDAData(
            vol_FDA_L=vol_1, dosis_FDA_g_hL=dosis_1, fecha_FDA=fecha_1,
            vol_FDA_2_L=0.0, dosis_FDA_2_g_hL=0.0, fecha_FDA_2=pd.NaT,
            horas_post_FDA_2_h=0.0,
            densidad_objetivo_FDA_2=np.nan
        )

    vol_2 = second.get("vol_L", 0.0)
    dosis_2 = fix_fda_dose(second.get("dosis_g_hL", 0.0), threshold=100.0)
    fecha_2 = second.get("fecha", pd.NaT)

    horas_post = compute_hours_diff_with_window(fecha_1, fecha_2, extra_window_h=12.0)

    dens_obj_2 = second.get("densidad_objetivo", np.nan)

    return FDAData(
        vol_FDA_L=vol_1, dosis_FDA_g_hL=dosis_1, fecha_FDA=fecha_1,
        vol_FDA_2_L=vol_2, dosis_FDA_2_g_hL=dosis_2, fecha_FDA_2=fecha_2,
        horas_post_FDA_2_h=horas_post,
        densidad_objetivo_FDA_2=dens_obj_2
    )


# =========================
# EXTRACCIÓN: INSUMOS OPERACIONALES
# =========================
def extract_insumos_operacionales(path_excel: str) -> InsumosData:
    df_ins = pd.read_excel(path_excel, sheet_name="Insumos Operacionales")

    def vol_keyword_1right(keyword: str) -> float:
        r, c = find_first_match(df_ins, keyword)
        val = df_ins.iat[r, c + 1] if (c + 1) < df_ins.shape[1] else np.nan
        valf = to_float(val)
        return 0.0 if (np.isnan(valf) or valf <= 0) else valf

    vol_sang = vol_keyword_1right("Sangría")
    vol_freek = vol_keyword_1right("Free K")
    vol_agua = vol_keyword_1right("Agua vegetal")
    vol_conc = vol_keyword_1right("Mosto concentrado")

    try:
        r_tart, c_tart = find_first_match(df_ins, "Ácido tartárico")
        unit_raw = df_ins.iat[r_tart, c_tart + 2] if (c_tart + 2) < df_ins.shape[1] else ""
        if is_unit_L(unit_raw):
            val_raw = df_ins.iat[r_tart, c_tart + 1] if (c_tart + 1) < df_ins.shape[1] else np.nan
            val = to_float(val_raw)
            vol_tart = 0.0 if (np.isnan(val) or val <= 0) else val
        else:
            vol_tart = 0.0
    except Exception:
        vol_tart = 0.0

    fda = extract_fda_complex_from_excel(path_excel)

    r_lev, c_lev = find_first_match(df_ins, "Levadura")
    vol_lev = to_float(df_ins.iat[r_lev, c_lev + 1]) if (c_lev + 1) < df_ins.shape[1] else 0.0
    vol_lev = 0.0 if (np.isnan(vol_lev) or vol_lev <= 0) else vol_lev

    tipo_inoc_raw = df_ins.iat[r_lev, c_lev + 9] if (c_lev + 9) < df_ins.shape[1] else ""
    tipo_inoculo = str(tipo_inoc_raw).strip()
    if tipo_inoculo.lower() == "nan":
        tipo_inoculo = ""

    pop_raw = to_float(df_ins.iat[r_lev, c_lev + 10]) if (c_lev + 10) < df_ins.shape[1] else 0.0
    if np.isnan(pop_raw) or pop_raw <= 0:
        pop = 0.0
    else:
        pop = pop_raw * 1e6 if pop_raw < 1e6 else pop_raw

    return InsumosData(
        vol_sang_L=vol_sang,
        vol_freek_L=vol_freek,
        vol_tart_L=vol_tart,
        vol_agua_L=vol_agua,
        vol_conc_L=vol_conc,
        vol_levadura_L=vol_lev,
        poblacion_levadura_cel_mL=pop,
        tipo_inoculo=tipo_inoculo,   # NUEVO
        fda=fda
    )


# =========================
# EXTRACCIÓN: PROV SENSORES
# =========================
def extract_prov_sensores(path_excel: str) -> ProvSensoresData:
    sheet_sensores = "Prov Sensores"
    col_time = "indice_tiempo_dias"
    col_dens = "densidad"
    col_tm = "temp_mosto"
    col_ts = "temp_sombrero"
    col_sp = "temp_setpoint"

    df_sens = pd.read_excel(path_excel, sheet_name=sheet_sensores)
    df_sens = df_sens[[col_time, col_dens, col_tm, col_ts, col_sp]].copy()

    for c in df_sens.columns:
        df_sens[c] = pd.to_numeric(df_sens[c], errors="coerce")

    df_sens = df_sens.dropna(subset=[col_time]).sort_values(col_time).reset_index(drop=True)
    df_sens["temp_promedio_raw"] = df_sens[[col_tm, col_ts]].mean(axis=1)

    t_h = df_sens[col_time].to_numpy(dtype=float) * 24.0

    return ProvSensoresData(
        t_h=t_h,
        densidad=df_sens[col_dens].to_numpy(dtype=float),
        temp_mosto=df_sens[col_tm].to_numpy(dtype=float),
        temp_sombrero=df_sens[col_ts].to_numpy(dtype=float),
        temp_setpoint=df_sens[col_sp].to_numpy(dtype=float),
        temp_promedio_raw=df_sens["temp_promedio_raw"].to_numpy(dtype=float),
    )


# =========================
# FUNCIÓN
# =========================
def load_fermentation_data(path_excel: str) -> FermentationData:
    ant = extract_antecedentes(path_excel)
    lab = extract_laboratorio(path_excel)
    ins = extract_insumos_operacionales(path_excel)
    sens = extract_prov_sensores(path_excel)

    return FermentationData(
        antecedentes=ant,
        laboratorio=lab,
        insumos=ins,
        sensores=sens
    )
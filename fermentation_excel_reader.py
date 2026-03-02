# -*- coding: utf-8 -*-
"""Lectura y cálculo de variables de fermentación desde un Excel.

La función principal `extraer_variables_fermentacion` replica la lógica del
notebook `Lectura_excel_fermentación.ipynb`: abre el archivo, localiza los
datos relevantes en las hojas y calcula:

- Concentración inicial de azúcar (g/L)
- YAN inicial (mg/L)
- Concentración inicial de biomasa en el reactor (g/L)
- Información de una posible segunda dosis de FDA (volumen,
  aporte en YAN y horas después de la primera adición)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _to_float(value: object) -> float:
    """Convierte strings con coma decimal a float; devuelve NaN en caso de fallo."""
    if isinstance(value, str):
        value = value.replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_match(df: pd.DataFrame, keyword: str) -> Tuple[int, int]:
    """Ubica la primera celda que contenga la palabra clave (búsqueda case-insensitive)."""
    mask = df.astype(str).apply(
        lambda col: col.str.contains(keyword, case=False, na=False))
    rows, cols = np.where(mask.values)
    if len(rows) == 0:
        raise ValueError(f'No se encontró "{keyword}" en la hoja de Excel.')
    return int(rows[0]), int(cols[0])


def _value_at_offset(df: pd.DataFrame, keyword: str, col_offset: int, row_offset: int = 0, default: float = 0.0,
    positive_only: bool = False,) -> float:
    """Devuelve el valor en la posición relativa a la primera aparición del keyword."""
    base_row, base_col = _first_match(df, keyword)
    row = base_row + row_offset
    col = base_col + col_offset
    try:
        raw = df.iat[row, col]
    except Exception:
        return default
    value = _to_float(raw)
    if positive_only and (pd.isna(value) or value <= 0):
        return 0.0
    if pd.isna(value):
        return default
    return value


def _extract_tartaric(df_insumos: pd.DataFrame) -> float:
    row, col = _first_match(df_insumos, "Ácido tartárico")
    unidad = str(df_insumos.iat[row, col + 2]).strip()
    if unidad == "(L)":
        return _to_float(df_insumos.iat[row, col + 1])
    return 0.0


def _extract_yeast(df_insumos: pd.DataFrame) -> Tuple[float, float]:
    row, col = _first_match(df_insumos, "Levadura")
    volumen = _value_at_offset(df_insumos, "Levadura", col_offset=1, positive_only=True)
    poblacion_raw = _to_float(df_insumos.iat[row, col + 10]) if volumen > 0 else 0.0
    if pd.isna(poblacion_raw) or poblacion_raw <= 0:
        poblacion = 0.0
    elif poblacion_raw < 1e6:
        poblacion = poblacion_raw * 1e6
    else:
        poblacion = poblacion_raw
    return volumen, poblacion


def _extract_fda_entries(df_insumos: pd.DataFrame):
    """Localiza todas las filas de FDA y separa la inicial de una posible dosis tardía (volumen o kg)."""
    mask = df_insumos.astype(str).apply(
        lambda col: col.str.contains("FDA", case=False, na=False)
    )
    rows, cols = np.where(mask.values)
    entries = []
    for r, c in zip(rows, cols):
        if c + 2 >= df_insumos.shape[1]:
            continue
        unidad = str(df_insumos.iat[r, c + 2]).strip().lower()
        es_volumen = unidad in {"(l)", "l"}
        es_kg = unidad in {"(kg)", "kg"}
        if not (es_volumen or es_kg):
            continue
        volumen = _to_float(df_insumos.iat[r, c + 1]) if es_volumen else 0.0
        if pd.isna(volumen) or volumen <= 0:
            volumen = 0.0
        dosis = _to_float(df_insumos.iat[r, c + 4])
        if pd.isna(dosis):
            dosis = 0.0
        fecha = (
            pd.to_datetime(df_insumos.iat[r, c + 7], errors="coerce", dayfirst=True)
            if c + 7 < df_insumos.shape[1]
            else pd.NaT
        )
        densidad = (
            _to_float(df_insumos.iat[r, c + 8]) if c + 8 < df_insumos.shape[1] else np.nan
        )
        density_ok = not pd.isna(densidad) and densidad >= 1000
        entries.append(
            {
                "row": int(r),
                "col": int(c),
                "volumen": volumen,
                "dosis": dosis,
                "fecha": fecha,
                "densidad": densidad,
                "density_ok": density_ok,
            }
        )

    if not entries:
        return {
            "principal": {"volumen": 0.0, "dosis": 0.0, "fecha": pd.NaT},
            "adicional": {
                "volumen": 0.0,
                "dosis": 0.0,
                "fecha": pd.NaT,
                "densidad": np.nan,
                "density_ok": False,
            },
        }

    entries.sort(key=lambda e: e["row"])
    principal = entries[0]
    adicional = next((e for e in entries[1:] if e["density_ok"]), None)
    if adicional is None:
        adicional = {
            "volumen": 0.0,
            "dosis": 0.0,
            "fecha": pd.NaT,
            "densidad": np.nan,
            "density_ok": False,
        }
    return {"principal": principal, "adicional": adicional}


def volumen_total(
    vol_mosto: float,
    vol_sang: float,
    vol_tart: float,
    vol_freek: float,
    vol_agua: float,
    vol_conc: float,
    vol_fda: float,
) -> Tuple[float, float]:
    vol_sangria = vol_mosto - vol_sang
    vol_final = vol_sangria + vol_tart + vol_freek + vol_agua + vol_conc + vol_fda
    return vol_sangria, vol_final


def calculo_conc_azucar_inicial(
    volumen: float, vol_sangria: float, vol_conc: float, brix: float, C_conc: float = 850.0
) -> float:
    conc_azucar = 12 * brix - 40
    masa_azucar = vol_sangria * conc_azucar + vol_conc * C_conc
    return masa_azucar / volumen


def calculo_conc_YAN_inicial(
    volumen: float, vol_sangria: float, YAN_0: float, dosis_FDA: float
) -> float:
    FDA_agregado = dosis_FDA * 25 / 10
    ajuste_YAN_vol = vol_sangria * YAN_0
    return (ajuste_YAN_vol + FDA_agregado * volumen) / volumen


def calculo_conc_biomasa_inicial(
    volumen: float, vol_levadura: float, poblacion_levadura: float
) -> Tuple[float, float]:
    # 1e-11 g -> 1 célula de levadura
    concentracion_biomasa = poblacion_levadura * 1e-11 * 1000  # g/L
    masa_biomasa = concentracion_biomasa * vol_levadura
    return concentracion_biomasa, masa_biomasa / volumen


def extraer_variables_fermentacion(
    excel_path: str | Path, concentracion_mosto_conc: float = 850.0
) -> Tuple[float, float, float, float, float, float]:
    """Lee el Excel y devuelve:

    (azúcar g/L, YAN mg/L, biomasa en reactor g/L,
     volumen FDA tardía (L), YAN FDA tardía (mg/L), horas post-FDA).
    """
    excel_path = Path(excel_path)

    df_ant = pd.read_excel(excel_path, sheet_name="Antecedentes")
    df_lab = pd.read_excel(excel_path, sheet_name="Laboratorio")
    df_ins = pd.read_excel(excel_path, sheet_name="Insumos Operacionales")

    vol_mosto = _to_float(df_ant.loc[0, "ant_vino_estimado_l"])
    brix_0 = _value_at_offset(df_lab, "Brix", col_offset=4)
    YAN_0 = _value_at_offset(df_lab, "YAN", col_offset=4)

    vol_sang = _value_at_offset(df_ins, "Sangría", col_offset=1, positive_only=True)
    vol_agua = _value_at_offset(df_ins, "Agua vegetal", col_offset=1, positive_only=True)
    vol_conc = _value_at_offset(df_ins, "Mosto concentrado", col_offset=1, positive_only=True)
    vol_tart = _extract_tartaric(df_ins)
    vol_freek = _value_at_offset(df_ins, "Free K", col_offset=1, positive_only=True)
    fda_entries = _extract_fda_entries(df_ins)
    dosis_FDA = fda_entries["principal"]["dosis"]
    vol_fda_principal = fda_entries["principal"].get("volumen", 0.0)
    vol_levadura, poblacion_levadura = _extract_yeast(df_ins)

    vol_sangria, vol_final = volumen_total(
        vol_mosto, vol_sang, vol_tart, vol_freek, vol_agua, vol_conc, vol_fda_principal
    )

    conc_azucar = calculo_conc_azucar_inicial(
        vol_final, vol_sangria, vol_conc, brix_0, C_conc=concentracion_mosto_conc
    )
    conc_YAN_mgL = calculo_conc_YAN_inicial(vol_final, vol_sangria, YAN_0, dosis_FDA)

    _, conc_biomasa_reactor = calculo_conc_biomasa_inicial(
        vol_final, vol_levadura, poblacion_levadura
    )

    vol_fda_extra = (
        fda_entries["adicional"]["volumen"] if fda_entries["adicional"]["density_ok"] else 0.0
    )
    dosis_fda_extra = (
        fda_entries["adicional"]["dosis"] if fda_entries["adicional"]["density_ok"] else 0.0
    )
    yan_fda_extra_mgL = dosis_fda_extra * 25 / 10 if fda_entries["adicional"]["density_ok"] else 0.0

    fecha_fda_base = fda_entries["principal"].get("fecha", pd.NaT)
    fecha_fda_extra = (
        fda_entries["adicional"].get("fecha", pd.NaT)
        if fda_entries["adicional"]["density_ok"]
        else pd.NaT
    )
    dias_post_fda = 0.0
    if not (pd.isna(fecha_fda_base) or pd.isna(fecha_fda_extra)):
        delta = fecha_fda_extra - fecha_fda_base
        dias_post_fda = max(delta.total_seconds() / 86400, 0.0)

    return (
        conc_azucar,
        conc_YAN_mgL,
        conc_biomasa_reactor,
        vol_fda_extra,
        yan_fda_extra_mgL,
        dias_post_fda,
    )

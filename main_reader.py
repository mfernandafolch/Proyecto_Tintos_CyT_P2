# -*- coding: utf-8 -*-
"""Procesa todos los Excel en una carpeta y calcula variables de fermentación."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Permite importar el módulo local aunque se ejecute desde otra carpeta
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fermentation_excel_reader import extraer_variables_fermentacion


def listar_excels(base_dir: Path):
    """Devuelve la lista de archivos Excel dentro de la carpeta (y subcarpetas)."""
    extensiones = {".xls", ".xlsx", ".xlsm"}
    return sorted(
        p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensiones
    )


def procesar_carpeta(base_dir: Path, salida_csv: Path):
    """Calcula azúcar, YAN, biomasa y FDA tardía para cada Excel y guarda un CSV resumen."""
    excels = listar_excels(base_dir)
    print(f"Encontrados {len(excels)} archivos Excel en {base_dir}")

    filas = []
    for excel in excels:
        try:
            (
                azucar,
                yan,
                biomasa,
                vol_fda_extra,
                yan_fda_extra,
                dias_post_fda,
            ) = extraer_variables_fermentacion(excel)
            filas.append(
                {
                    "archivo": str(excel),
                    "azucar_gL": azucar,
                    "YAN_mgL": yan,
                    "biomasa_reactor_gL": biomasa,
                    "FDA_extra_volumen_L": vol_fda_extra,
                    "FDA_extra_YAN_mgL": yan_fda_extra,
                    "dias_post_FDA": dias_post_fda,
                    "error": "",
                }
            )
            print(f"OK  -> {excel.name}")
        except Exception as exc:  # noqa: BLE001
            filas.append(
                {
                    "archivo": str(excel),
                    "azucar_gL": None,
                    "YAN_mgL": None,
                    "biomasa_reactor_gL": None,
                    "FDA_extra_volumen_L": None,
                    "FDA_extra_YAN_mgL": None,
                    "dias_post_FDA": None,
                    "error": str(exc),
                }
            )
            print(f"FAIL -> {excel.name}: {exc}")

    pd.DataFrame(filas).to_csv(salida_csv, index=False)
    print(f"\nResultados guardados en: {salida_csv}")


if __name__ == "__main__":
    carpeta = Path(
        r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Datos históricos\2024\MA 24 LOU"
    )
    salida = HERE / "resultados_MA24LOU.csv"
    procesar_carpeta(carpeta, salida)

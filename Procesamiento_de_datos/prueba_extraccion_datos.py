from extraccion_datos import load_fermentation_data
import numpy as np

def print_extraction_summary(data):
    """
    Imprime un resumen claro y estructurado de la información extraída.
    """

    print("\n" + "="*70)
    print("RESUMEN DE EXTRACCIÓN DE DATOS")
    print("="*70)

    # -----------------------------------
    # ANTECEDENTES
    # -----------------------------------
    print("\n--- ANTECEDENTES ---")
    print(f"Volumen mosto estimado (L): {data.antecedentes.vol_mosto_est_L}")
    print(f"Brix inicial: {data.antecedentes.brix_inicial}")

    # -----------------------------------
    # LABORATORIO
    # -----------------------------------
    print("\n--- LABORATORIO ---")
    print(f"YAN inicial (mg/L): {data.laboratorio.yan0_mgL}")
    print(f"Alcohol final (% v/v): {data.laboratorio.alcohol_grado}")
    print(f"Etanol final (g/L): {data.laboratorio.E_final_obs_gL}")

    # -----------------------------------
    # INSUMOS
    # -----------------------------------
    print("\n--- INSUMOS OPERACIONALES ---")
    print(f"Sangría (L): {data.insumos.vol_sang_L}")
    print(f"Free K (L): {data.insumos.vol_freek_L}")
    print(f"Ácido tartárico (L): {data.insumos.vol_tart_L}")
    print(f"Agua vegetal (L): {data.insumos.vol_agua_L}")
    print(f"Mosto concentrado (L): {data.insumos.vol_conc_L}")

    print("\nLevadura:")
    print(f"  Volumen (L): {data.insumos.vol_levadura_L}")
    print(f"  Población (cel/mL): {data.insumos.poblacion_levadura_cel_mL}")

    # -----------------------------------
    # FDA
    # -----------------------------------
    print("\n--- FDA ---")

    fda = data.insumos.fda

    print("Primera adición:")
    print(f"  Volumen (L): {fda.vol_FDA_L}")
    print(f"  Dosis (g/hL): {fda.dosis_FDA_g_hL}")
    print(f"  YAN aporte (mg/L): {fda.yan_FDA_mgL}")
    print(f"  Fecha: {fda.fecha_FDA}")

    print("\nSegunda adición:")
    print(f"  Volumen (L): {fda.vol_FDA_2_L}")
    print(f"  Dosis (g/hL): {fda.dosis_FDA_2_g_hL}")
    print(f"  YAN aporte (mg/L): {fda.yan_FDA_2_mgL}")
    print(f"  Fecha: {fda.fecha_FDA_2}")
    print(f"  Horas post 1ª adición: {fda.horas_post_FDA_2_h}")

    # -----------------------------------
    # SENSORES
    # -----------------------------------
    print("\n--- PROV SENSORES ---")

    t = data.sensores.t_h
    dens = data.sensores.densidad
    temp = data.sensores.temp_promedio_raw

    print(f"Número de puntos: {len(t)}")

    if len(t) > 0:
        print(f"Tiempo inicial (h): {t[0]:.2f}")
        print(f"Tiempo final (h): {t[-1]:.2f}")
        print(f"Duración total (h): {t[-1] - t[0]:.2f}")

        if np.all(np.isnan(dens)):
            print("Densidad: todos los valores son NaN")
        else:
            dens_clean = dens[~np.isnan(dens)]
            print(f"Densidad inicial válida: {dens_clean[0]:.4f}")
            print(f"Densidad final válida: {dens_clean[-1]:.4f}")
            print(f"Densidad min/max: {np.min(dens_clean):.4f} / {np.max(dens_clean):.4f}")

        print(f"Temp promedio min/max (°C): {np.nanmin(temp):.2f} / {np.nanmax(temp):.2f}")

    print("\n" + "="*70)

path_excel = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Datos históricos\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx"

data = load_fermentation_data(path_excel)

print_extraction_summary(data)
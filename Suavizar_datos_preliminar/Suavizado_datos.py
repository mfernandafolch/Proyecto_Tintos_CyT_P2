"""
SUAVIZADO DE DATOS EXPERIMENTALES CON SPLINES
Y REDUCCIÓN EXPLÍCITA DEL NÚMERO DE PUNTOS
------------------------------------------

Pipeline:
1) Lee datos experimentales desde Excel
2) Estima ruido efectivo (método de residuos)
3) Ajusta smoothing splines (funciones continuas)
4) Evalúa splines cada 1 hora hasta el fin de los datos
5) Grafica:
   - Original vs spline
   - Original vs puntos horarios
6) Imprime tamaños de los sets finales
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from pathlib import Path


# =========================================================
# 0) CARGA DEL ARCHIVO EXCEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

excel_filename = "Data CS 25 P. VALDES estanque 219.xlsx"
# excel_filename = "Data CS 24 LOU estanque 58.xlsx"

excel_path = DATA_DIR / excel_filename

sheet_name = "Prov Sensores"

col_t   = "indice_tiempo_dias"
col_dens = "densidad"
col_tm  = "temp_mosto"
col_ts  = "temp_sombrero"
col_sp  = "temp_setpoint"

def densidad_to_sugar(densidad):
    """Convierte densidad (g/L) a concentración de azúcar (g/L) usando fórmula empírica."""
    return 2.5616 * densidad - 2577.4


# =========================================================
# 1) PARÁMETROS GLOBALES
# =========================================================

# Para estimar ruido (suavizado MUY fuerte)
noise_window_temp = 81
noise_window_dens = 51
noise_poly = 1

# Factores de suavidad del spline
alpha_temp = 2.0
alpha_dens = 2.0


# =========================================================
# 2) LECTURA Y LIMPIEZA DE DATOS
# =========================================================

if not excel_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {excel_path}")

df_raw = pd.read_excel(str(excel_path), sheet_name=sheet_name)

df = df_raw[[col_t, col_dens, col_tm, col_ts, col_sp]].copy()

for c in [col_t, col_dens, col_tm, col_ts, col_sp]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[col_t]).sort_values(col_t).reset_index(drop=True)

# Temperatura promedio (RAW)
df["temp_promedio_raw"] = df[[col_tm, col_ts]].mean(axis=1)

t = (df[col_t].to_numpy())*24 # Convertir días a horas


# =========================================================
# 3) ESTIMACIÓN DEL RUIDO (MÉTODO DE RESIDUOS)
# =========================================================

def estimate_noise_residual(y, window, poly):
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    yv = y[mask]

    if len(yv) < window:
        return np.nan

    y_smooth = savgol_filter(yv, window, poly)
    residuals = yv - y_smooth

    return np.std(residuals, ddof=1)


sigma_temp = estimate_noise_residual(df[col_tm], noise_window_temp, noise_poly)
sigma_dens = estimate_noise_residual(df[col_dens], noise_window_dens, noise_poly)

N_temp = np.sum(~df[col_tm].isna())
N_dens = np.sum(~df[col_dens].isna())

spline_s_temp = alpha_temp * N_temp * sigma_temp**2
spline_s_dens = alpha_dens * N_dens * sigma_dens**2

print("----- PARÁMETROS SPLINE -----")
print(f"σ temperatura ≈ {sigma_temp:.3f} °C, s temperatura = {spline_s_temp:.2f}")
print(f"σ densidad    ≈ {sigma_dens:.3f} g/L, s densidad = {spline_s_dens:.2f}")
print("-----------------------------\n")


# =========================================================
# 4) AJUSTE DE SPLINES
# =========================================================

def fit_spline(t, y, s, k=2):
    mask = ~np.isnan(y)
    return UnivariateSpline(t[mask], y[mask], s=s, k=k)


spline_dens = fit_spline(t, df[col_dens].to_numpy(), spline_s_dens)
spline_tm  = fit_spline(t, df[col_tm].to_numpy(),  spline_s_temp)
spline_ts  = fit_spline(t, df[col_ts].to_numpy(),  spline_s_temp)
spline_tp  = fit_spline(t, df["temp_promedio_raw"].to_numpy(), spline_s_temp)

# Evaluación spline en tiempos originales
dens_spline_full = spline_dens(t)
tm_spline_full   = spline_tm(t)
ts_spline_full   = spline_ts(t)
tp_spline_full   = spline_tp(t)
# Conversión a azúcar (g/L)
azucar_raw = densidad_to_sugar(df[col_dens])
azucar_spline_full = densidad_to_sugar(dens_spline_full)


# =========================================================
# 5) SUBMUESTREO EXPLÍCITO (1 h) — RESPETA INICIO DENSIDAD
# =========================================================

t_start_density_hours = df.loc[df[col_dens].notna(), col_t].iloc[0] * 24

N_original = len(t)
# Vector horario desde el primer dato de densidad hasta el final
t_reduced = np.arange(t_start_density_hours, t.max() + 1e-6, 1.0)
if t_reduced[-1] < t.max():
    t_reduced = np.append(t_reduced, t.max())
N_reduced = len(t_reduced)

dens_reduced = spline_dens(t_reduced)
tm_reduced   = spline_tm(t_reduced)
ts_reduced   = spline_ts(t_reduced)
tp_reduced   = spline_tp(t_reduced)
azucar_reduced = densidad_to_sugar(dens_reduced)

print("----- TAMAÑO DE LOS SETS -----")
print(f"Puntos originales : {N_original}")
print(f"Puntos reducidos  : {N_reduced}")
print("-----------------------------\n")


# =========================================================
# 6A) GRÁFICO: DATOS ORIGINALES vs CURVA SPLINE
# =========================================================

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(t, azucar_raw, "--", color="tab:blue", alpha=0.5, label="azúcar raw (g/L)")
ax1.plot(t, azucar_spline_full, "-", color="tab:blue", lw=2.5, label="azúcar spline (g/L)")

ax2.plot(t, df[col_tm], "--", color="tab:red", alpha=0.5, label="temp_mosto raw")
ax2.plot(t, tm_spline_full, "-", color="tab:red", lw=2, label="temp_mosto spline")

ax2.plot(t, df[col_ts], "--", color="tab:green", alpha=0.5, label="temp_sombrero raw")
ax2.plot(t, ts_spline_full, "-", color="tab:green", lw=2, label="temp_sombrero spline")

ax2.plot(t, df["temp_promedio_raw"], "--", color="tab:orange", alpha=0.5, label="temp_promedio raw")
ax2.plot(t, tp_spline_full, "-", color="tab:orange", lw=2, label="temp_promedio spline")

ax2.plot(t, df[col_sp], "--", color="gold", alpha=0.6, label="set point")

ax1.set_xlabel("Tiempo (horas)")
ax1.set_ylabel("Azúcar (g/L)")
ax2.set_ylabel("Temperatura (°C)")
ax1.grid(True, alpha=0.3)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=3)

plt.title("Datos originales vs curva spline")
plt.tight_layout()
plt.show()


# =========================================================
# 6B) GRÁFICO: DATOS ORIGINALES vs PUNTOS SUBMUESTREADOS
# =========================================================

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(t, azucar_raw, "--", color="tab:blue", alpha=0.5, label="azúcar raw (g/L)")
ax1.plot(t_reduced, azucar_reduced, ".", color="tab:blue", ms=6, label="azúcar reducida (g/L)")

ax2.plot(t, df[col_tm], "--", color="tab:red", alpha=0.5, label="temp_mosto raw")
ax2.plot(t_reduced, tm_reduced, ".", color="tab:red", ms=6, label="temp_mosto reducida")

ax2.plot(t, df[col_ts], "--", color="tab:green", alpha=0.5, label="temp_sombrero raw")
ax2.plot(t_reduced, ts_reduced, ".", color="tab:green", ms=6, label="temp_sombrero reducida")

ax2.plot(t, df["temp_promedio_raw"], "--", color="tab:orange", alpha=0.5, label="temp_promedio raw")
ax2.plot(t_reduced, tp_reduced, ".", color="tab:orange", ms=6, label="temp_promedio reducida")

ax2.plot(t, df[col_sp], "--", color="gold", alpha=0.6, label="set point")

ax1.set_xlabel("Tiempo (horas)")
ax1.set_ylabel("Azúcar (g/L)")
ax2.set_ylabel("Temperatura (°C)")
ax1.grid(True, alpha=0.3)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=3)

plt.title("Datos originales vs puntos submuestreados (1 h)")
plt.tight_layout()
plt.show()
# =========================================================

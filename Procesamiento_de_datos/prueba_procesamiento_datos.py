import numpy as np
import matplotlib.pyplot as plt

from procesamiento_datos import process_excel

# path_excel = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 PAROT+AURORA estanque 54.xlsx"
# path_excel = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx"
path_excel = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Proyecto_Tintos_CyT_P2\Datos_industriales\CS\100.000 L\Data CS 24 LOU estanque 54.xlsx"

out = process_excel(
    path_excel=path_excel,
    t_muestreo_h=3.0,   # cada cuántas horas muestrear
    t_end_h=None
)

# =========================
# Imprimir condiciones iniciales
# =========================
init = out.init
print("\n" + "=" * 60)
print("CONDICIONES INICIALES (g/L)")
print("=" * 60)
print(f"X0 (Biomasa):      {init.X0_gL:.6g}")
print(f"N0 (Nitrógeno):    {init.N0_gL:.6g}" if not np.isnan(init.N0_gL) else "N0 (Nitrógeno):    NaN")
print(f"G0 (Glucosa):      {init.G0_gL:.6g}" if not np.isnan(init.G0_gL) else "G0 (Glucosa):      NaN")
print(f"F0 (Fructosa):     {init.F0_gL:.6g}" if not np.isnan(init.F0_gL) else "F0 (Fructosa):     NaN")
print(f"E0 (Etanol):       {init.E0_gL:.6g}")
print(f"E_final_obs (g/L): {init.E_final_obs_gL:.6g}" if not np.isnan(init.E_final_obs_gL) else "E_final_obs (g/L): NaN")
print("=" * 60)

# =========================
# Resumen mínimo FDA2 / Nadd (la grilla manda)
# =========================
meta = out.meta
print("\n" + "=" * 60)
print("FDA2 / Nadd (RESUMEN MÍNIMO)")
print("=" * 60)

t_excel_abs = meta.get("FDA2_t_fecha_abs_h", np.nan)
t_corr_abs  = meta.get("FDA2_t_evento_corregido_abs_h", np.nan)  # continuo corregido
t_pulse_abs = meta.get("Nadd_event_time_abs_h", np.nan)          # el que cae en la grilla
t_pulse_rel = meta.get("Nadd_event_time_rel_h", np.nan)
idx_pulse   = meta.get("Nadd_event_index", np.nan)
val_pulse   = meta.get("Nadd_value_gL", 0.0)

def _fmt(x, nd=2):
    return "NaN" if (x is None or np.isnan(x)) else f"{float(x):.{nd}f}"

print(f"t_muestreo_h:                 {_fmt(meta.get('t_muestreo_h', np.nan), nd=2)} h")
print(f"t_start_opt_abs_h:            {_fmt(meta.get('t_start_opt_abs_h', np.nan), nd=2)} h")
print(f"FDA2_t_fecha_abs_h (Excel):   {_fmt(t_excel_abs, nd=2)} h")
print(f"FDA2_t_corregido_abs_h:       {_fmt(t_corr_abs, nd=2)} h")
print(f"Nadd aplicado EN GRILLA abs:  {_fmt(t_pulse_abs, nd=2)} h")
print(f"Nadd aplicado EN GRILLA rel:  {_fmt(t_pulse_rel, nd=2)} h") 
print(f"Nadd index (en grilla):       {int(idx_pulse) if not np.isnan(idx_pulse) else -1}")
print(f"Nadd valor (g/L):             {_fmt(val_pulse, nd=6)}")
print("=" * 60)

# =========================
# Series para graficar
# =========================
t_rel = np.asarray(out.profiles.t_rel_h, dtype=float)

azucar_total = np.asarray(out.profiles.azucar, dtype=float)

temp_mosto = np.asarray(out.profiles.temp_mosto, dtype=float)
temp_sombrero = np.asarray(out.profiles.temp_sombrero, dtype=float)
temp_prom = np.asarray(out.profiles.temp_promedio, dtype=float)
temp_setpoint = np.asarray(out.profiles.setpoint, dtype=float)

Nadd = np.asarray(out.profiles.Nadd_gL, dtype=float)

mask_sugar = ~np.isnan(t_rel) & ~np.isnan(azucar_total)
mask_tm = ~np.isnan(t_rel) & ~np.isnan(temp_mosto)
mask_ts = ~np.isnan(t_rel) & ~np.isnan(temp_sombrero)
mask_tp = ~np.isnan(t_rel) & ~np.isnan(temp_prom)
mask_setp = ~np.isnan(t_rel) & ~np.isnan(temp_setpoint)
mask_nadd = ~np.isnan(t_rel) & ~np.isnan(Nadd)

# =========================
# Plot: Azúcar + Temperaturas + Nadd (solo lo necesario)
# =========================
fig, ax1 = plt.subplots(figsize=(11, 5))

# Azúcar (eje izq)
ax1.plot(t_rel[mask_sugar], azucar_total[mask_sugar], "o-", label="Azúcar total (g/L)")
if (not np.isnan(init.G0_gL)) and (not np.isnan(init.F0_gL)):
    ax1.scatter(0.0, init.G0_gL + init.F0_gL, color="red", zorder=5, label="Azúcar inicial (t=0)")
ax1.axvline(0.0, linestyle="--", linewidth=1, label="Inicio optimizado (t=0)")
ax1.set_xlabel("Tiempo desde inicio optimizado (h)")
ax1.set_ylabel("Azúcar total suavizada (g/L)")
ax1.grid(True)

# Línea del pulso (EL que realmente importa)
if not np.isnan(t_pulse_rel):
    ax1.axvline(float(t_pulse_rel), linestyle=(0, (6, 2)), linewidth=2.5, label="Nadd (pulso en grilla)")

# Temperaturas (eje der)
ax2 = ax1.twinx()
ax2.plot(t_rel[mask_tm], temp_mosto[mask_tm], "*:", label="Temp Mosto (°C)")
ax2.plot(t_rel[mask_ts], temp_sombrero[mask_ts], "*:", label="Temp Sombrero (°C)")
ax2.plot(t_rel[mask_tp], temp_prom[mask_tp], "*:", linewidth=2, label="Temp Promedio (°C)")
ax2.plot(t_rel[mask_setp], temp_setpoint[mask_setp], "-", alpha=0.5, label="Setpoint (°C)")
ax2.set_ylabel("Temperatura (°C)")

# Nadd como stem (opcional, pero útil)
ax3 = None
if np.any(Nadd[mask_nadd] > 0):
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.10))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    ax3.stem(t_rel, Nadd, linefmt="k-", markerfmt="ko", basefmt=" ")
    ax3.set_ylabel("Nadd (g/L)")
    ax3.set_ylim(0, max(1e-6, 1.2 * float(np.nanmax(Nadd))))

# Leyenda combinada
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
if ax3 is not None:
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="best", fontsize="small")
else:
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")

plt.title("Azúcar total, temperaturas y pulso Nadd (t=0 inicio optimizado)")
plt.tight_layout()
plt.show()

print(f"\nHay {len(t_rel)} puntos en el perfil.")
print("=" * 60)
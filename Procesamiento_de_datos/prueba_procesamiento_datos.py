from py_compile import main

import numpy as np
import matplotlib.pyplot as plt

from procesamiento_datos import process_excel

path_excel = r"C:\Users\MARIA\OneDrive - Universidad Católica de Chile\Escritorio\Concha y Toro\Datos históricos\CS\51.700 L\Data CS 25 SUC. IVAN VALDES estanque 239.xlsx"


out = process_excel(path_excel=path_excel,
        t_muestreo_h=3.0,   # cada cuántas horas muestrear
        t_end_h=None)

# print(out.init)
# print(out.profiles)
# print(out.meta)

# =========================
# Imprimir condiciones iniciales
# =========================
init = out.init
print("\n" + "=" * 60)
print("CONDICIONES INICIALES (g/L)")
print("=" * 60)
print(f"X0 (Biomasa):      {init.X0_gL}")
print(f"N0 (Nitrógeno):    {init.N0_gL}")
print(f"G0 (Glucosa):      {init.G0_gL}")
print(f"F0 (Fructosa):     {init.F0_gL}")
print(f"E0 (Etanol):       {init.E0_gL}")
print(f"E_final_obs (g/L): {init.E_final_obs_gL}")
print("=" * 60)

# =========================
# Tiempo relativo al inicio optimizado
# =========================
t_abs = out.profiles.t_abs_h
t_rel = out.profiles.t_rel_h   # ya viene con t=0 en inicio optimizado

# Azúcar total suavizada
azucar_total = out.profiles.azucar

# Temperaturas
temp_mosto = out.profiles.temp_mosto
temp_sombrero = out.profiles.temp_sombrero
temp_prom = out.profiles.temp_promedio
temp_setpoint = out.profiles.setpoint

# Nadd (g/L) impulso por 2ª FDA
Nadd = out.profiles.Nadd_gL

# Máscaras NaN
mask_sugar = ~np.isnan(t_rel) & ~np.isnan(azucar_total)
mask_tm = ~np.isnan(t_rel) & ~np.isnan(temp_mosto)
mask_ts = ~np.isnan(t_rel) & ~np.isnan(temp_sombrero)
mask_tp = ~np.isnan(t_rel) & ~np.isnan(temp_prom)
mask_setp = ~np.isnan(t_rel) & ~np.isnan(temp_setpoint)
mask_nadd = ~np.isnan(t_rel) & ~np.isnan(Nadd)

# =========================
# Plot con doble eje (y Nadd como evento)
# =========================
fig, ax1 = plt.subplots(figsize=(10, 5))

# Eje 1: Azúcar
ax1.plot(t_rel[mask_sugar], azucar_total[mask_sugar], "o-", label="Azúcar total (g/L)")
ax1.scatter(0, init.G0_gL + init.F0_gL, color="red", zorder=5, label="Azúcar total inicial (t=0)")
ax1.set_xlabel("Tiempo desde inicio optimizado (h)")
ax1.set_ylabel("Azúcar total suavizada (g/L)")
ax1.grid(True)

# Línea vertical en inicio
ax1.axvline(0.0, linestyle="--", linewidth=1, label="Inicio optimizado (t=0)")

# Marcar evento de adición FDA2 (si existe)
if np.any(Nadd[mask_nadd] > 0):
    idx_event = int(np.argmax(Nadd))   # único impulso
    t_event = float(t_rel[idx_event])
    ax1.axvline(t_event, linestyle=":", linewidth=2, label="Adición FDA 2 (evento)")

# Eje 2: Temperaturas
ax2 = ax1.twinx()
ax2.plot(t_rel[mask_tm], temp_mosto[mask_tm], "*:", label="Temp Mosto (°C)")
ax2.plot(t_rel[mask_ts], temp_sombrero[mask_ts], "*:", label="Temp Sombrero (°C)")
ax2.plot(t_rel[mask_tp], temp_prom[mask_tp], "*:", label="Temp Promedio (°C)")
ax2.plot(t_rel[mask_setp], temp_setpoint[mask_setp], "y-", alpha=0.5)
ax2.set_ylabel("Temperatura (°C)")

# NUEVO: graficar Nadd como "impulso" en el eje derecho (escala separada)
# Para que se vea aunque sea pequeño, lo dibujamos como stem y en el eje derecho.
if np.any(Nadd[mask_nadd] > 0):
    ax3 = ax1.twinx()
    # mover el tercer eje un poquito a la derecha
    ax3.spines["right"].set_position(("axes", 1.08))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)

    markerline, stemlines, baseline = ax3.stem(
        t_rel, Nadd, linefmt="k-", markerfmt="ko", basefmt=" "
    )
    ax3.set_ylabel("Nadd (g/L)")
    ax3.set_ylim(0, max(1e-6, 1.2 * np.nanmax(Nadd)))

# Leyenda combinada (ax1 + ax2 + ax3 si existe)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

if "ax3" in locals():
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="best", fontsize="small")
else:
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")

plt.title("Azúcar total suavizada, temperaturas y evento Nadd (t=0 en inicio optimizado)")
plt.tight_layout()
plt.show()


print(f'Hay {len(t_rel)} puntos de datos en el perfil de tiempo relativo.')
print("=" * 60)
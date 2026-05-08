from procesamiento_datos import process_excel
import matplotlib.pyplot as plt

excel_path = r"C:\Users\p-mfolch\Documents\Proyecto_Tintos_CyT\Datos_industriales\CS\52.400 L\Data CS 25 EL BOLDO estanque 133.xlsx"
t_muestreo = 3.0

data_excel = process_excel(path_excel=excel_path, t_muestreo_h = t_muestreo)
init = data_excel.init


def fmt_gL(value):
	return "N/A" if value is None or value != value else f"{value:.2f} g/L"

# Graficar datos procesados con segundo eje para densidad
profiles = data_excel.profiles
t_dias = profiles.t_rel_h / 24

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.set_title('Datos procesados para CS 25 EL BOLDO estanque 133.')  # Título de la gráfica

line_tm, = ax1.plot(t_dias, profiles.temp_mosto, linestyle='-', color='#BB126D', alpha = 0.2, label='Temperatura Mosto (°C)')
line_ts, = ax1.plot(t_dias, profiles.temp_sombrero, linestyle='-', color='#4B0C2D',  alpha = 0.2, label='Temperatura Sombrero (°C)')
line_tp, = ax1.plot(t_dias, profiles.temp_promedio, '*-', linewidth=2, color='m', label='Temperatura Promedio (°C)')
# line_sp, = ax1.plot(t_dias, profiles.setpoint, linestyle='-', color='#EF8A00', label='Setpoint Temperatura (°C)')

line_dens, = ax2.plot(t_dias, profiles.azucar, '*-', color='#00F2FE',  alpha = 0.2 ,label='Azúcar (g/L)')

ax1.set_xlabel('Tiempo (días)')
ax1.set_ylabel('Temperatura (°C)')
ax2.set_ylabel('Azúcar (g/L)')

# lines = [line_tm, line_ts, line_tp, line_sp, line_dens]
lines = [line_tm, line_ts, line_tp, line_dens]

labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')
plt.grid(True)

texto_init = (
	"Condiciones iniciales: "
	f"$X_0$={fmt_gL(init.X0_gL)} | "
	f"$N_0$={fmt_gL(init.N0_gL)} | "
	f"$G_0$={fmt_gL(init.G0_gL)} | "
	f"$F_0$={fmt_gL(init.F0_gL)} | "
	f"$E_0$={fmt_gL(init.E0_gL)} | "
	f"$E_{{\\text{{final}}}}$={fmt_gL(init.E_final_obs_gL)}"
)

fig.text(0.5, 0.04, texto_init, ha='center', va='bottom', fontsize=10)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()

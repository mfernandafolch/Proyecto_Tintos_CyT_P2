
# Calibración y validación (optimizacion_pymoo)

Esta carpeta agrupa los scripts de calibración (ajuste de parámetros) y validación empleados con PSO de `pymoo` y otros métodos del proyecto. Esta carpeta funciona como la unión de las fases de calibración y validación.

Estructura y propósito

- `main_ajustes_random.py`: ejecuta estimaciones aleatorias (busca parámetros por muestreo aleatorio). Útil como baseline y para explorar el espacio de parámetros sin algoritmo de optimización.
- `main_pymoo.py`: ejecuciones con PSO de`pymoo` para optimización/ajuste (si está presente y configurado).
- `main_plot_std_bands_sugar.py`: genera gráficos de bandas de desviación estándar (validación visual de ajuste sobre datos de azúcar/serie temporal correspondiente).
- `main_plot_uncertainty.py`: genera gráficos de incertidumbre con simulaciones de Monte Carlo de los parámetros en el rango de variabilidad de las predicciones. 
- `plot_ajuste.py`: utilidades para visualizar ajustes (ajustes vs datos observados).


Requisitos

Configura un entorno con las librerías habituales del proyecto. Ejemplo mínimo:

```bash
conda create -n mi_env python=3.10 -y
conda activate mi_env
pip install numpy scipy pandas matplotlib seaborn pymoo
```

Ejecución

- Ejecución de calibración por muestreo aleatorio:

```bash
conda activate mi_env
python main_ajustes_random.py
```

- Ejecutar validación/plots:

```bash
python main_plot_std_bands_sugar.py
python main_plot_uncertainty.py
```

Salida esperada

- Los scripts de calibración suelen guardar archivos con las mejores configuraciones y registros de ejecución (JSON, CSV o pickles) en subcarpetas.
- Los scripts de validación generan figuras (PNG/SVG) y pueden guardar datos de bandas/incertidumbre en CSV para análisis posterior.


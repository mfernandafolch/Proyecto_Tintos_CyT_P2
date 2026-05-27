
# Calibración y validación, Modelo Coleman

Descripción

Esta carpeta contiene los experimentos de calibración y validación realizados sobre el modelo Coleman. Aquí se agrupan las ejecuciones de ajuste (incluyendo estimaciones aleatorias) y los scripts para generar visualizaciones de validación e incertidumbre.

Archivos principales

- `main_ajustes_random.py`: muestreo aleatorio de parámetros (baseline y exploración del espacio de búsqueda).
- `main_pymoo.py`: orquestador para correr optimizaciones con PSO de `pymoo` sobre el modelo Coleman.
- `pymoo_opt_coleman.py`: implementación específica de la función objetivo, la optimización con PSO y envoltorio para `pymoo`.
- `plot_ajuste.py`: utilidades para visualizar ajustes (predicción vs datos observados).
- `main_plot_std_bands_sugar.py`: genera gráficas de bandas de desviación estándar (validación visual sobre series de azúcar u otras señales relevantes).
- `main_plot_uncertainty.py`: gráficos de incertidumbre/variabilidad (ensambles, bandas, análisis de sensibilidad).
- `resultados_buenos/`, `otros_resultados/`: carpetas con salidas de experimentos (mejores runs y resultados complementarios).

Requisitos

Instala las dependencias del proyecto en tu entorno (ejemplo mínimo):

```bash
conda create -n mi_env python=3.10 -y
conda activate mi_env
pip install numpy scipy pandas matplotlib seaborn pymoo
```

Ejecución — ejemplos rápidos

- Ejecutar ajustes aleatorios:

```bash
conda activate mi_env
python main_ajustes_random.py
```

- Ejecutar optimización con `pymoo`:

```bash
python main_pymoo.py
```

- Generar gráficos de validación e incertidumbre:

```bash
python main_plot_std_bands_sugar.py
python main_plot_uncertainty.py
```

Salida esperada

- Los scripts de ajuste guardan registros de parámetros evaluados y métricas de desempeño; revisa `resultados_buenos/` y `otros_resultados/` para ejemplos.
- Los scripts de validación generan figuras y, opcionalmente, archivos CSV con las bandas/valores de incertidumbre.


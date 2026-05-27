
# Modelo Zenteno, Códigos

Resumen

Esta carpeta contiene el código fuente asociado al modelo dinámico de Zenteno: extracción y procesamiento de datos, modelo, simulaciones y folders con pruebas de optimización y ajuste de parámetros.

Estructura principal

- `extraccion_datos.py`: scripts y funciones para leer y preparar datos de entrada desde los archivos en `Datos_industriales/`.
- `procesamiento_datos.py`: transformación y limpieza de series temporales y variables necesarias para ajustar el modelo.
- `plot_data.py`: utilidades para visualización básica de datos observados.
- `modelo_dinamico.py`: definición del modelo dinámico (ecuaciones, parámetros y funciones de integración).
- `simulacion.py` / `Simulacion_solve_ivp.ipynb`: scripts y notebook para correr simulaciones (uso de `solve_ivp` u otros integradores).
- `main_procesamiento_y_simulacion.py`: script principal que corre el pipeline de procesamiento y simulación con parámetros por defecto.
- `codigos_optimizacion/`: carpeta con submódulos y experimentos de optimización y ajuste de parámetros (tuning, validación, comparaciones). Revisa sus subcarpetas: `hiperparametros/`, `optimizacion_pymoo/`, `codigos_de_prueba_opti/`.

Uso rápido

1. Crear y activar entorno (ejemplo con conda):

```bash
conda create -n mi_env python=3.10 -y
conda activate mi_env
pip install numpy scipy pandas matplotlib
# paquetes adicionales según subcarpetas: pymoo, pyswarms, scikit-learn, etc.
```

2. Preprocesar datos y ejecutar simulación de ejemplo:

```bash
conda activate mi_env
python main_procesamiento_y_simulacion.py
```

3. Para ajustes y validación revisa `codigos_optimizacion/` y sus README locales (cada subcarpeta tiene instrucciones concretas).


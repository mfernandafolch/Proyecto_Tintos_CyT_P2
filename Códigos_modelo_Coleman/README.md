
# Modelo Coleman, Código

Resumen

Esta carpeta contiene el código fuente para el modelo dinámico de Coleman: extracción y procesamiento de datos, definición del modelo, simulaciones y una subcarpeta con experimentos de optimización y ajuste.

Contenido principal

- `extraccion_datos.py`: lectura y preparación de datos de entrada.
- `procesamiento_datos.py`: transformaciones y limpieza de datos antes de simulación/ajuste.
- `plot_data.py`: funciones de visualización de series y datos experimentales.
- `modelo_dinamico_coleman.py`: implementación del modelo dinámico (ecuaciones y evaluación)  de Coleman.
- `simulacion_coleman.py`: scripts para ejecutar simulaciones del modelo con el modelo Coleman.
- `main_procesamiento_y_simulacion.py`: pipeline de ejemplo: procesa datos y corre simulaciones con parámetros por defecto.
- `optimizacion_pymoo/`: experimentos de calibración y validación (ajustes aleatorios, `pymoo`, plots de incertidumbre).

Requisitos

Configura un entorno con las dependencias básicas del proyecto. Ejemplo:

```bash
conda create -n mi_env python=3.10 -y
conda activate mi_env
pip install numpy scipy pandas matplotlib
# Añade pymoo, pyswarms o scikit-learn según los experimentos que quieras ejecutar
```

Uso rápido

1. Preprocesar datos y ejecutar una simulación de ejemplo:

```bash
conda activate mi_env
python main_procesamiento_y_simulacion.py
```

2. Para ajustes y validación: entra en `optimizacion_pymoo/` y sigue su README.


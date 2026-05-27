
# Hiperparámetros y tuning

Esta carpeta contiene los scripts y resultados relacionados con la búsqueda y validación de hiperparámetros para el optimizador usado en el proyecto (PSO de pymoo).

Propósito

- Probar y ajustar hiperparámetros de algoritmos de optimización mediante validación cruzada y experimentos sistemáticos.
- Guardar resultados (runs, mejores configuraciones, métricas) para análisis y selección de configuraciones finales.

Archivos principales

- `main_pso_cv_tuning.py` — orquestador para pruebas de PSO con validación cruzada y guardado de resultados.
- `pso_cv_tuning.py` — implementación de la búsqueda/validación de hiperparámetros para PSO.
- `pymoo_opt.py` — experimentos con `pymoo` (si se usa para optimización o tuning).

Carpetas de resultados

- `resultados_cv_pso_20260408_171225/` — ejemplo de carpeta con resultados de validación cruzada, contiene JSON/CSV con los mejores parámetros y métricas. Los resultados se abren con Excel.
- Otras carpetas de resultados similares pueden estar presentes en esta ruta; revisa el contenido para ver detalles de cada experimento.

Requisitos

Instala un entorno con las librerías necesarias. Ejemplo mínimo:

```bash
conda create -n mi_env python=3.10 -y
conda activate mi_env
pip install numpy scipy pandas scikit-learn matplotlib
# Paquetes específicos para tuning/optimizadores:
pip install pyswarms pymoo
```

Ejecución y uso

- Para lanzar un tuning de PSO (ejemplo):

```bash
conda activate mi_env
python main_pso_cv_tuning.py
```

- Revisa las variables/configuraciones en la cabecera de cada script para ajustar rangos de búsqueda, número de folds, semillas y rutas de salida.

Formato de resultados

- Los experimentos guardan: parámetros probados, métrica objetivo por fold, tiempos de ejecución y el mejor conjunto de hiperparámetros. 

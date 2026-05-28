# Proyecto Tintos CyT

Repositorio de trabajo para el procesamiento de datos industriales de fermentación y la simulación y ajuste de modelos dinámicos asociados al proceso de vino tinto. Este proyecto se enmarca en una práctica profesional realizada entre enero y mayo del 2026.

El proyecto organiza información real de fermentaciones industriales, la prepara para análisis y la usa como base para comparar, simular y calibrar modelos de fermentación alcohólica. A grandes rasgos, el trabajo apunta a representar variables como biomasa de levaduras, nitrógeno asimilable, azúcares y etanol bajo condiciones industriales.

## Qué se hizo en el proyecto

El desarrollo del repositorio se ha enfocado en:

- Reunir y organizar datos industriales por año, variedad y tipo de archivo;
- Construir pipelines de extracción y preprocesamiento de datos;
- Implementar modelos dinámicos de fermentación;
- Correr simulaciones sobre los datos procesados;
- Explorar estrategias de ajuste y optimización de parámetros;
- Generar gráficos para analizar ajuste, incertidumbre y bandas de variación.

## Estructura principal

### `Códigos_modelo_Coleman/`

Contiene la línea de trabajo asociada al modelo de Coleman.

Incluye scripts para:

- Extracción de datos industriales;
- Procesamiento y limpieza de datos;
- Visualización básica de series y resultados;
- Definición del modelo dinámico Coleman;
- Simulación del sistema;
- Ejecución de un flujo principal de procesamiento y simulación.

La subcarpeta `optimizacion_pymoo/` reúne experimentos de calibración y análisis, incluyendo ajustes aleatorios, optimización con PSO de `pymoo`, gráficos de ajuste y herramientas para estudiar incertidumbre y bandas de dispersión.

### `Códigos_modelo_Zenteno/`

Contiene la línea de trabajo asociada al modelo de Zenteno.

Esta carpeta agrupa:

- Scripts de extracción y procesamiento de datos;
- El modelo dinámico;
- Scripts y notebook de simulación;
- Un flujo principal de procesamiento y simulación;
- Una carpeta de experimentos de optimización y ajuste.

Dentro de `codigos_optimizacion/` se concentran pruebas con optimizadores, búsqueda de hiperparámetros, comparaciones entre enfoques y experimentos basados en PSO de `pymoo` y otros métodos de ajuste.

### `Datos_industriales/`

Contiene los datos industriales usados por el proyecto.

La organización principal es:

- Por año, en carpetas como `2024/` y `2025/`;
- Por variedad de uva, en carpetas como `CA/`, `CS/`, `MA/`, `ME/` y `SY/`;
- Por lote o conjunto de mediciones dentro de cada variedad.

Esta carpeta actúa como la fuente de datos brutos para el preprocesamiento, la simulación y la calibración de los modelos.

## Flujo general de trabajo

1. Se cargan los datos industriales desde `Datos_industriales/`.
2. Se limpian, transforman y estructuran para su uso en simulación.
3. Se ejecutan los modelos dinámicos desde las carpetas de Coleman o Zenteno.
4. Se comparan resultados observados y simulados.
5. Se exploran ajustes de parámetros y análisis de incertidumbre.


## Requisitos generales

Se recomienda usar **Python 3.10 o superior**.

Dependiendo de la carpeta o script, pueden requerirse bibliotecas como:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `pymoo`
- `pyswarms`
- `scikit-learn`

## Autor

María Fernanda Folch Díaz

Ingeniera Civil en Biotecnología, mención Procesos 

Pontificia Universidad Católica de Chile  

Proyecto desarrollado durante práctica profesional en **Centro de Investigación e Innovación (CII), Viña Concha y Toro**.

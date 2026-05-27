
# Pruebas de optimizadores

Esta carpeta contiene pruebas y scripts usados para comparar y experimentar con distintos algoritmos de optimización aplicados al modelo.

Resumen:

- **Objetivo:** validar y comparar el comportamiento de varios optimizadores sobre funciones objetivo del proyecto.
- **Optimizadores probados (según registro):** Dual Annealing (DA), Differential Evolution (DE), Particle Swarm Optimization (PSO).

Contenido de la carpeta

- `main2_opt_pso.py`: prueba/ejecución con PSO.
- `main2_opt.py`: script de prueba general (ajustes aleatorios / comparativos).
- `max_prueba_opt.py`: pruebas para funciones objetivo con máximo buscado.
- `prueba_opt.py`: script de pruebas básicas y ejemplos mínimos.
- `pso_opt.py`: implementación / experimento con PSO.
- `Resulados optimizadores.xlsx`: recolección de resultados obtenidos con pruebas de diferentes algoritmos y optimizadores. 

Requisitos

Instala las dependencias típicas del proyecto en tu entorno (por ejemplo `mi_env`):

```bash
conda create -n mi_env python=3.10 -y
conda activate mi_env
pip install numpy scipy matplotlib
# Instala según lo que use cada script: pymoo, pyswarms, etc.
pip install pymoo pyswarms
```

Ejecución

Ejemplos rápidos:

```bash
conda activate mi_env
python main2_opt.py
python pso_opt.py
python main2_opt_pso.py
```

Notas

- Revisa el encabezado de cada script para ver parámetros configurables (número de iteraciones, semillas, límites de búsqueda, funciones objetivo).
- Algunos scripts guardan resultados o trazas en archivos locales; otros imprimen resultados por consola. Adapta las rutas internas si quieres centralizar salidas.
- Si no recuerdas exactamente qué optimizador usó cada script, abre el archivo y busca las importaciones (`from scipy.optimize import dual_annealing`, `differential_evolution`, `pyswarms` o `pymoo`) para confirmarlo.


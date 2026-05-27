
# Datos industriales

Descripción

Esta carpeta contiene los datos industriales utilizados en el proyecto, organizados por año y por variedad de uva. Está pensada para almacenar los archivos de mediciones, controles y datos necesarios para el procesamiento y la calibración de los modelos.

Estructura principal

- `2024/` — datos del año 2024 organizados por variedad.
- `2025/` — datos del año 2025 organizados por variedad.
- Subcarpetas por variedad: `CA/`, `CS/`, `MA/`, `ME/`, `SY/` que corresponden a Carmenere, Cabernet Sauvignon, Malbec, Merlot y Syrah, respectivamente.

Notas sobre el contenido

- Cada subcarpeta por variedad contiene archivos de series temporales y mediciones del proceso (CSV / Excel o subcarpetas con mediciones por lote). Revisa cada carpeta para ver el formato exacto de los ficheros.
- El archivo `Inoculo.xlsx` contiene la revisión sobre la concentración inicial de levaduras cuando no aparece ese dato en el proceso. 

Formato y carga rápida (ejemplo)

Usa `pandas` para cargar los archivos; ejemplo genérico:

```python
import pandas as pd

# ejemplo: cargar un CSV de la variedad Cabernet Sauvignon 2024
df = pd.read_csv('2024/CS/mediciones_lote_001.csv')

# ejemplo: leer el archivo de inóculo (Excel)
inoc = pd.read_excel('inoculo.xlsx')
```


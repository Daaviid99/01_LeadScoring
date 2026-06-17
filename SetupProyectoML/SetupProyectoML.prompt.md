---

agent: agent

---



\# OBJETIVO

Configurar proyectos de Machine Learning con Python. Crearás una estructura completa y profesional siguiendo las siguientes instrucciones.



\# INSTRUCCIONES

Ejecuta los siguientes pasos en orden:



\## 1. CREAR ESTRUCTURA DE CARPETAS Y ARCHIVOS



Crea esta estructura exacta:

\- 01\_Documentos/

\- 02\_datos/01\_Originales/

\- 02\_datos/02\_Validacion/

\- 02\_datos/03\_Entrenamiento/

\- 02\_datos/04\_Caches/

\- 03\_notebooks/

\- 04\_scripts/

\- 05\_modelos/

\- 06\_resultados/

\- 07\_despliegue/

\- 99\_otros/

\- .github/prompts/

\- .github/agents/



Crea en la raíz del proyecto el archivo .gitignore con este contenido:

```

\# Archivos del sistema

desktop.ini

Thumbs.db

\# Entornos

.env

.venv/

venv/

\_\_pycache\_\_/

\*.py\[cod]

\# Jupyter

.ipynb\_checkpoints/

.notebooks/

\# Datos

datos/

\*.csv

\*.xlsx

\*.parquet

\*.feather

\*.db

\*.sqlite

\*.duckdb

\# Entregables

entregables/

reports/

\# Archivos de log

\*.log

\# Configuración de IDE

.vscode/

.idea/

```



Crea en la raíz del proyecto el archivo .github/copilot-instructions.md con este contenido:

```

\- Me responderás siempre en español de España.

\- Comenzarás cada mensaje con este emoji 🤖

\- Siempre genera código para el script abierto, nunca para la ventana interactiva, salvo que se indique explícitamente lo contrario.

\- La ventana interactiva solo se usará para verificar resultados o mensajes de error, pero no para generar código.

\- No expliques el código si no se solicita.

```



Crea el archivo .env vacío en la raíz del proyecto.



\## 2. CONFIGURAR ENTORNO CONDA



Ejecuta uno a uno estos comandos en la terminal integrada de VSCode.

Si alguno falla, detente y corrige el error antes de continuar.

Si no instala con conda, usa pip donde sea posible.

Si finalmente no puedes instalar alguna librería, déjala para más adelante y genera un aviso en la terminal al final del proceso indicando qué librerías no se han podido instalar.



```

conda create -n nombre\_proyecto python -y

conda activate nombre\_proyecto

conda install -y numpy pandas matplotlib seaborn scikit-learn jupyter notebook ipykernel streamlit python-dotenv joblib pathlib sqlalchemy requests openpyxl

pip install fastapi uvicorn xgboost plotly pyjanitor

python -m ipykernel install --user --name=nombre\_proyecto

```



IMPORTANTE: Reemplaza "nombre\_proyecto" con el nombre de la carpeta actual del proyecto.



\## 3. CREAR NOTEBOOK INICIAL



Crea el archivo 03\_notebooks/01\_Importacion\_Datos.ipynb con la primera celda conteniendo:



```

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from dotenv import load\_dotenv

import janitor

import os

from pathlib import Path

import openpyxl

import sqlalchemy as sa



\# Desactivar notación científica

pd.set\_option('display.float\_format', lambda x: '%.3f' % x)

np.set\_printoptions(suppress=True)



\# Cargar variables de entorno

load\_dotenv()



print("✅ Librerías importadas correctamente")

```



```


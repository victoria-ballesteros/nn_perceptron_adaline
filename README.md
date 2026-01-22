# Clasificación con Perceptrón vs Adaline – Análisis Comparativo

Proyecto de machine learning que implementa y compara dos algoritmos clásicos de clasificación lineal: **Perceptrón** y **Adaline**, utilizando un conjunto de datos de muestras biológicas.

## Estructura del Proyecto
. <br>
├── adapters/ # Capa de adaptadores para procesamiento de datos <br>
│ ├── get_training_set.py # Carga y preparación del dataset <br>
│ ├── linear_model.py # Implementación de modelos lineales <br>
│ ├── model_result_visualizer.py # Visualización de resultados <br>
│ ├── train.py # Entrenamiento de modelos <br>
│ ├── training_data_visualizer.py # Visualización de datos  <br>
│ └── utils.py # Funciones auxiliares <br>
│ <br>
├── core/ # Lógica principal del flujo <br>
│ └── execute.py # Orquestador de ejecución <br>
│ <br>
├── data/ # Datos y resultados <br>
│ ├── plots/ # Gráficos generados <br>
│ ├── models/ # Modelos guardados (opcional) <br>
│ ├── ports/ # Interfaces de entrada/salida <br>
│ ├── glass.csv # Dataset principal <br>
│ ├── glass.names # Descripción del dataset <br>
│ └── glass.tag # Etiquetas adicionales <br>
│ <br>
├── Dockerfile # Configuración del entorno contenedorizado <br>
├── main.py # Punto de entrada principal <br>
└── requirements.txt # Dependencias de Python <br>

## Ejecución Rápida

### 1. Construir la imagen Docker
```bash
docker build -t glass_ml .
```
### 2. Ejecutar el contenedor
```bash
docker run -it --rm -v $(pwd)/data:/app/data glass_ml
```
### 3. Dentro del contenedor, ejecutar:
```bash
python3 main.py
```

## Métricas Esperadas (ejemplo de salida)
- Perceptrón:
   Accuracy:  0.930
   Recall:    0.926
   Precision: 0.953
   F1-Score:  0.931

- Adaline:
   Accuracy:  0.939
   Recall:    0.941
   Precision: 0.982
   F1-Score:  0.941

- Error rate:
   Perceptrón: 0.0701
   Adaline:    0.0607

# Modelo Predictivo de Cancelaciones de Reservas Hoteleras

## Descripción
Este proyecto desarrolla un modelo de machine learning para predecir la probabilidad de cancelación de reservas de hotel, basándose en características de la reserva y del cliente. El objetivo es proporcionar a los hoteles una herramienta que les permita anticipar cancelaciones y optimizar su gestión de inventario y recursos.

## Estructura del Proyecto
```
├── modelo_reserva_hoteles.ipynb # Notebook principal con análisis exploratorio, preprocesamiento y modelado
├── paper/ # Documentación del proyecto
│   ├── paper.md # Documento principal del proyecto según lineamientos académicos
│   └── ST1613 ANTEPROYECTO.md # Guía del anteproyecto con los lineamientos a seguir
├── images/ # Visualizaciones generadas durante el análisis
│   ├── distribucion-booking-status.png
│   ├── mapa-de-calor.png
│   ├── cancelacion-por-mercado.png
│   └── ... # Otras visualizaciones
├── data/ # Datos del proyecto
│   └── Hotel Reservations.csv # Conjunto de datos de reservas hoteleras (obtenido de Kaggle)
└── README.md # Este archivo
```


## Metodología
El proyecto sigue una metodología estructurada de aprendizaje automático:
1. Análisis exploratorio de datos (EDA)
2. Preprocesamiento (normalización, codificación, balanceo de clases)
3. Entrenamiento y evaluación de modelos (Regresión Logística, Random Forest, Gradient Boosting)
4. Optimización de hiperparámetros
5. Interpretación de resultados

## Conjunto de Datos
El conjunto de datos contiene información sobre reservas hoteleras, incluyendo:
- Detalles del cliente
- Características de la reserva (lead time, tipo de habitación, etc.)
- Historial del cliente
- Variable objetivo: booking_status (cancelada o no cancelada)

## Hallazgos Principales
- Variables como lead_time, número de solicitudes especiales y huéspedes repetidos son predictores importantes
- Existen patrones de cancelación por segmento de mercado y estacionalidad
- El modelo logra identificar con precisión las reservas con alta probabilidad de cancelación

## Tecnologías Utilizadas
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Imbalanced-learn (SMOTE)

## Estado del Proyecto
En desarrollo - Actualmente en fase de preprocesamiento de datos

## Autores
- [Tu Nombre]

# Predicción de la Demanda de Taxis en New York City

Este proyecto utiliza un modelo de machine learning para predecir la demanda de taxis en la ciudad de Nueva York basándose en datos históricos y condiciones climáticas actuales. Fue desarrollado utilizando Python y Streamlit para una interfaz interactiva.

## Ejecutar la App en Streamlit

La App ha sido desplegada utilizando Streamlit Cloud. Puedes acceder en el siguiente enlace:

[https://pi2-dataanalysis-internet.streamlit.app/](https://prediccion-demanda-taxis-nyc.streamlit.app/)

## Descripción del Proyecto

La aplicación permite a los usuarios:
- Predecir la demanda diaria de taxis para una fecha seleccionada.
- Visualizar la distribución horaria de la demanda basada en predicciones históricas.
- Consultar la demanda por distritos en un día específico.

## Funcionalidades
1. **Predicción de Demanda**: El modelo utiliza datos históricos y climáticos para realizar predicciones precisas de la demanda diaria de taxis.
2. **Visualización Interactiva**:
   - Gráfico de barras para la distribución horaria de la demanda.
   - Tabla interactiva con la demanda por distrito.
3. **Integración con Open Meteo API**: Obtención de datos climáticos actuales para ajustar las predicciones.

## Estructura del Proyecto
- `app.py`: Script principal que contiene la lógica de la aplicación Streamlit.
- `machine_learning_model.py`: Código para el entrenamiento y validación del modelo de machine learning.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `taxi_demand_model.joblib`: Modelo de machine learning entrenado.
- `taxis_boroughdemand.parquet` y `taxis_hourlydemand.parquet`: Datasets utilizados para las predicciones.

## Requisitos
Para ejecutar el proyecto, necesitas instalar las dependencias enumeradas en `requirements.txt`

## Cómo Ejecutar
1. **Clona este repositorio**

2. **Instala las dependencias**
   
3. **Ejecuta la aplicación**

## Visualizaciones
 -**Distribución Horaria:** Gráficos interactivos muestran la demanda horaria estimada.
 
 -**Demanda por Distritos:** Tablas dinámicas con datos específicos de cada distrito. 
   
## Dataset
El modelo utiliza datasets en formato Parquet que contienen:

 -**Demanda histórica por horas y distritos.** 
 
 -**Condiciones climáticas para cada día.**

## Contribución
Contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request con tus sugerencias.

## Contacto
Desarrollado por Leonardo Renteria. Si tienes preguntas, no dudes en contactarme: Email: leo.rencalderon@gmail.com



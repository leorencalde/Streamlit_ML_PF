import streamlit as st
import pandas as pd
import joblib
import datetime
import openmeteo_requests
import matplotlib.pyplot as plt
from datetime import timedelta

# Cargar el modelo de machine learning
model = joblib.load('taxi_demand_model.joblib')

# Cargar el dataset histórico de demanda por horas
df_historico = pd.read_parquet('taxis_hourlydemand.parquet')

# Cargar el dataset de demanda por boroughs
df_borough_demand = pd.read_parquet('taxis_boroughdemand.parquet')

# Convertir la columna 'date' de índice a columna normal
df_borough_demand = df_borough_demand.reset_index()

# Convertir la columna 'date' del DataFrame a datetime si no lo está
df_borough_demand['date'] = pd.to_datetime(df_borough_demand['date'])

# Asegurarse de que el DataFrame `df_historico` tenga la columna `dayofweek`
if 'dayofweek' not in df_historico.columns:
    df_historico['dayofweek'] = pd.to_datetime(df_historico['date']).dt.dayofweek

# Diccionario con descripciones de códigos del clima
weather_code_descriptions = {
    0: "Despejado",
    1: "Principalmente despejado",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Niebla",
    48: "Escarcha",
    51: "Llovizna ligera",
    53: "Llovizna moderada",
    55: "Llovizna densa",
    56: "Llovizna helada ligera",
    57: "Llovizna helada densa",
    61: "Lluvia ligera",
    63: "Lluvia moderada",
    65: "Lluvia intensa",
    66: "Lluvia helada ligera",
    67: "Lluvia helada intensa",
    71: "Nevadas ligeras",
    73: "Nevadas moderadas",
    75: "Nevadas intensas",
    77: "Granizo",
    80: "Chubascos ligeros",
    81: "Chubascos moderados",
    82: "Chubascos intensos",
    85: "Chubascos de nieve ligeros",
    86: "Chubascos de nieve intensos",
    95: "Tormenta eléctrica ligera",
    96: "Tormenta eléctrica con granizo ligero",
    99: "Tormenta eléctrica con granizo intenso"
}

# Función para obtener datos climáticos
def obtener_datos_climaticos(date_str):
    days_ahead = (datetime.datetime.strptime(date_str, '%Y-%m-%d').date() - datetime.date.today()).days
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.7128,
        "longitude": -74.006,
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max"],
        "timezone": "America/New_York",
        "forecast_days": days_ahead + 1
    }
    responses = openmeteo_requests.Client().weather_api(url, params=params)
    
    if responses and len(responses) > 0:
        response = responses[0]
        daily = response.Daily()
        return {
            "weather_code": daily.Variables(0).ValuesAsNumpy()[days_ahead],
            "temperature_2m_max": round(daily.Variables(1).ValuesAsNumpy()[days_ahead], 1),
            "temperature_2m_min": round(daily.Variables(2).ValuesAsNumpy()[days_ahead], 1),
            "precipitation_sum": round(daily.Variables(3).ValuesAsNumpy()[days_ahead], 1),
            "rain_sum": round(daily.Variables(4).ValuesAsNumpy()[days_ahead], 1),
            "snowfall_sum": round(daily.Variables(5).ValuesAsNumpy()[days_ahead], 1),
            "precipitation_hours": round(daily.Variables(6).ValuesAsNumpy()[days_ahead], 1),
            "wind_speed_10m_max": round(daily.Variables(7).ValuesAsNumpy()[days_ahead], 1),
        }
    return None

# Función para encontrar el día más parecido en el histórico
def encontrar_dia_parecido(dia_semana, df_historico):
    # Filtrar el dataset histórico por el día de la semana
    df_filtrado = df_historico[df_historico['dayofweek'] == dia_semana]
    
    # Agrupar por día y obtener la demanda total diaria
    df_agrupado = df_filtrado.groupby('date').sum().reset_index()
    
    # Encontrar el día con la demanda más cercana a la predicción total
    return df_filtrado[df_filtrado['date'] == df_agrupado.iloc[(df_agrupado['demand'] - st.session_state['total_demand']).abs().argsort()[:1]]['date'].values[0]]

# Función para mostrar la demanda por distritos en una tabla
def mostrar_tabla_distritos(fecha_filtrada):
    # Filtrar los datos del dataset de demanda por boroughs por la fecha seleccionada
    df_filtered = df_borough_demand[df_borough_demand['date'] == fecha_filtrada]

    # Eliminar la columna 'date' y el índice
    df_filtered = df_filtered.drop(columns=['date']).reset_index(drop=True)
    
    # Mostrar la tabla en Streamlit sin la columna 'date' y sin el índice
    st.subheader("Demanda por Boroughs")
    st.dataframe(df_filtered)

# Título de la aplicación
st.title("Predicción de Demanda de Taxis en New York City")
st.subheader("By Datavision")

# Obtener la fecha de hoy
today = datetime.date.today()

# Entrada del usuario, permitiendo seleccionar hasta dos días después de hoy
date = st.date_input("Fecha", today, min_value=today, max_value=today + timedelta(days=2))
date_str = date.strftime('%Y-%m-%d')

# Botón para obtener datos climáticos y hacer la predicción
if st.button("Obtener datos climáticos y predecir demanda"):
    clima = obtener_datos_climaticos(date_str)
    
    if clima:
        # Almacenar los datos climáticos y la predicción en el estado de la sesión
        st.session_state['clima'] = clima
        st.session_state['dayofweek'] = date.weekday()

        # Crear el DataFrame con los datos de entrada para la predicción
        input_data = pd.DataFrame([{
            'weather_code': clima['weather_code'],
            'temperature_2m_max': clima['temperature_2m_max'],
            'temperature_2m_min': clima['temperature_2m_min'],
            'rain_sum': clima['rain_sum'],
            'snowfall_sum': clima['snowfall_sum'],
            'precipitation_hours': clima['precipitation_hours'],
            'wind_speed_10m_max': clima['wind_speed_10m_max'],
            'day': date.day,
            'month': date.month,
            'dayofweek': date.weekday(),
            'season_number': 1 if date.month in [12, 1, 2] else 2 if date.month in [3, 4, 5] else 3 if date.month in [6, 7, 8] else 4
        }])

        # Realizar la predicción
        prediction = model.predict(input_data)
        total_demand = int(prediction[0])

        # Almacenar la predicción en el estado de la sesión
        st.session_state['total_demand'] = total_demand

        # Encontrar el día más parecido en el histórico y asignar la fecha filtrada
        df_similar_day = encontrar_dia_parecido(st.session_state['dayofweek'], df_historico)
        st.session_state['fecha_filtrada'] = df_similar_day['date'].iloc[0]

# Mostrar los datos climáticos si ya se han predicho
if 'clima' in st.session_state:
    clima = st.session_state['clima']

    # Obtener la descripción del código del clima
    weather_description = weather_code_descriptions.get(clima["weather_code"], "Código desconocido")

    # Formatear los valores para mostrar solo un decimal
    temp_max = f"{clima['temperature_2m_max']:.1f}"
    temp_min = f"{clima['temperature_2m_min']:.1f}"
    precip_sum = f"{clima['precipitation_sum']:.1f}"
    precip_hours = f"{clima['precipitation_hours']:.1f}"
    wind_speed = f"{clima['wind_speed_10m_max']:.1f}"

    # Mostrar los resultados con métricas y formato gráfico
    st.subheader("Datos Climáticos Obtenidos:")
    st.metric(label="Descripción del Clima", value=weather_description)
    st.metric(label="Temperatura Máxima (°C)", value=temp_max)
    st.metric(label="Temperatura Mínima (°C)", value=temp_min)
    st.metric(label="Precipitación Total (mm)", value=precip_sum)
    st.metric(label="Horas de Precipitación", value=precip_hours)
    st.metric(label="Velocidad Máxima del Viento (km/h)", value=wind_speed)

    # Mostrar la predicción con un tamaño de fuente grande y efecto dinámico
    st.markdown(
        f"<h1 style='text-align: center; color: blue; font-size: 38px; font-weight: bold; animation: blink 1.5s infinite;'>"
        f"La predicción de demanda de taxis es: {st.session_state['total_demand']} viajes.</h1>",
        unsafe_allow_html=True
    )

    # CSS para animación de parpadeo
    st.markdown(
        """
        <style>
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Verificar si ya se ha predicho la demanda y permitir mostrar la gráfica
if 'total_demand' in st.session_state:
    # Botón para mostrar la gráfica de demanda por horas
    if st.button("Mostrar Demanda por Horas"):
        total_demand = st.session_state['total_demand']
        dia_semana = st.session_state['dayofweek']

        # Encontrar el día más parecido en el histórico
        df_similar_day = encontrar_dia_parecido(dia_semana, df_historico)
        fecha_filtrada = st.session_state['fecha_filtrada']
        
        # Calcular la distribución horaria basada en el día más similar
        distribucion_horaria = df_similar_day.groupby('hour')['demand'].sum()
        distribucion_horaria_normalizada = distribucion_horaria / distribucion_horaria.sum()

        # Ajustar la demanda horaria basada en la predicción total
        demanda_por_hora = distribucion_horaria_normalizada * total_demand

        # Graficar la demanda por horas
        fig, ax = plt.subplots()
        demanda_por_hora.plot(kind='bar', ax=ax)
        ax.set_xlabel('Hora del Día')
        ax.set_ylabel('Demanda de Taxis')
        ax.set_title('Distribución de la Demanda de Taxis por Hora')
        st.pyplot(fig)

    # Botón para mostrar la demanda por distritos
    if st.button("Mostrar demanda por distritos"):
        mostrar_tabla_distritos(st.session_state['fecha_filtrada'])




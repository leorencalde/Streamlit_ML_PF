import streamlit as st
import pandas as pd
import joblib
import datetime
import openmeteo_requests

# Cargar el modelo de machine learning
model = joblib.load('taxi_demand_model.joblib')

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
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.7128,
        "longitude": -74.006,
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max"],
        "timezone": "America/New_York",
        "forecast_days": 1
    }
    responses = openmeteo_requests.Client().weather_api(url, params=params)
    
    if responses and len(responses) > 0:
        response = responses[0]
        daily = response.Daily()
        return {
            "weather_code": daily.Variables(0).ValuesAsNumpy()[0],
            "temperature_2m_max": round(daily.Variables(1).ValuesAsNumpy()[0], 1),
            "temperature_2m_min": round(daily.Variables(2).ValuesAsNumpy()[0], 1),
            "precipitation_sum": round(daily.Variables(3).ValuesAsNumpy()[0], 1),
            "rain_sum": round(daily.Variables(4).ValuesAsNumpy()[0], 1),
            "snowfall_sum": round(daily.Variables(5).ValuesAsNumpy()[0], 1),
            "precipitation_hours": round(daily.Variables(6).ValuesAsNumpy()[0], 1),
            "wind_speed_10m_max": round(daily.Variables(7).ValuesAsNumpy()[0], 1),
        }
    return None

# Título de la aplicación
st.title("Predicción de Demanda de Taxis en New York City")

# Obtener la fecha de hoy
today = datetime.date.today()

# Entrada del usuario
date = st.date_input("Fecha", today, min_value=today, max_value=today)
date_str = date.strftime('%Y-%m-%d')

# Botón para obtener datos climáticos y hacer la predicción
if st.button("Obtener datos climáticos y predecir demanda"):
    clima = obtener_datos_climaticos(date_str)
    
    if clima:
        st.subheader("Datos Climáticos Obtenidos:")

        # Obtener la descripción del código del clima
        weather_description = weather_code_descriptions.get(clima["weather_code"], "Código desconocido")

        # Formatear los valores con un decimal antes de mostrarlos
        temp_max = f"{clima['temperature_2m_max']:.1f}"
        temp_min = f"{clima['temperature_2m_min']:.1f}"
        precip_sum = f"{clima['precipitation_sum']:.1f}"
        precip_hours = f"{clima['precipitation_hours']:.1f}"
        wind_speed = f"{clima['wind_speed_10m_max']:.1f}"

        # Mostrando resultados con métricas y formato gráfico
        st.metric(label="Descripción del Clima", value=weather_description)
        st.metric(label="Temperatura Máxima (°C)", value=temp_max)
        st.metric(label="Temperatura Mínima (°C)", value=temp_min)
        st.metric(label="Precipitación Total (mm)", value=precip_sum)
        st.metric(label="Horas de Precipitación", value=precip_hours)
        st.metric(label="Velocidad Máxima del Viento (km/h)", value=wind_speed)

        # Crear el DataFrame con los datos de entrada para la predicción
        input_data = pd.DataFrame([{
            'weather_code': clima['weather_code'],
            'temperature_2m_max': float(temp_max),
            'temperature_2m_min': float(temp_min),
            'rain_sum': clima['rain_sum'],
            'snowfall_sum': clima['snowfall_sum'],
            'precipitation_hours': float(precip_hours),
            'wind_speed_10m_max': float(wind_speed),
            'day': date.day,
            'month': date.month,
            'dayofweek': date.weekday(),
            'season_number': 1 if date.month in [12, 1, 2] else 2 if date.month in [3, 4, 5] else 3 if date.month in [6, 7, 8] else 4
        }])

        # Realizar la predicción
        prediction = model.predict(input_data)

        # Mostrar la predicción con un tamaño de fuente grande y efecto dinámico
        st.markdown(
            f"<h1 style='text-align: center; color: blue; font-size: 38px; font-weight: bold; animation: blink 1.5s infinite;'>"
            f"La predicción de demanda de taxis para hoy es: {int(prediction[0])} viajes.</h1>",
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
    else:
        st.error("Error al obtener datos climáticos.")




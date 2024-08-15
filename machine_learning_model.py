# machine_learning_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib

# Cargar los datasets
df_taxis = pd.read_parquet('./taxis_dailydemand.parquet')
df_clima = pd.read_parquet('./weather_daily.parquet')

# Asegúrate de que ambas columnas 'date' estén en el formato datetime
df_taxis['date'] = pd.to_datetime(df_taxis['date'])
df_clima['date'] = pd.to_datetime(df_clima['date'])

# Combinar datasets basados en la fecha y hora
df = pd.merge(df_taxis, df_clima, on='date')
print(df.head())

# Extraer características de la fecha
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek

# Variables y features
features = ['season_number','weather_code', 'temperature_2m_max', 'temperature_2m_min', 'rain_sum', 'snowfall_sum', 'wind_speed_10m_max', 'day', 'month', 'dayofweek']
target = 'demand'

# División en entrenamiento y prueba
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['season_number','weather_code', 'temperature_2m_max', 'temperature_2m_min', 'rain_sum', 'snowfall_sum', 'wind_speed_10m_max']),
        ('cat', OneHotEncoder(), ['day', 'month', 'dayofweek'])
    ])

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lgb.LGBMRegressor(n_estimators=100, random_state=42))
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Guardar el modelo entrenado
joblib.dump(pipeline, './taxi_demand_model.joblib')

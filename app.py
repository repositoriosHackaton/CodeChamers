
import pandas as pd
from pandas import DataFrame
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime

def save_dataset(symbol) -> DataFrame:
    data = yf.download(symbol, start='2024-01-16')

    return data 


# Configurar Streamlit
st.title("Predicción de Tendencia del mercado de acciones")

option = st.selectbox(
    "Company Stocks",
    ("AAPL", 
     "MSFT", 
     "TSLA", 
     "META", 
     "NVDA", 
     "GOOG",
     "NFLX"
))

df = save_dataset(option)
df.reset_index(inplace=True)

df['Weekly_Return'] = df['Close'].pct_change(5)
df['Weekly_Trend'] = np.where(df['Weekly_Return'] > 0, 1, 0)
df = df.dropna()

for i in range(1, 5):
    df[f'WeekReturn{i}'] = df['Weekly_Return'].shift(i)
df = df.dropna()

# Filtrar los datos hasta el 1 de julio de 2024
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] <= '2024-07-01']

features = ['WeekReturn1', 'WeekReturn2', 'WeekReturn3', 'WeekReturn4']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Cargar el modelo
model = load_model('trendaizer.h5')

# Hacer predicciones
y_pred_proba = model.predict(X_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int)

# Crear DataFrame de comparación
comparison_df = pd.DataFrame({
    'Date': df['Date'],
    'Actual_Trend': df['Weekly_Trend'],
    'Predicted_Trend': y_pred.flatten(),
    'Predicted_Probability': y_pred_proba.flatten()
})

# Filtrar solo las fechas de fin de semana
comparison_df = comparison_df[comparison_df['Date'].dt.weekday == 4]  # 4 es el índice para viernes
comparison_df = comparison_df.sort_values('Date')

# Predicción para la próxima semana (semana del 8 de julio)
last_week_data = df.iloc[-1][features].values.reshape(1, -1)
last_week_scaled = scaler.transform(last_week_data)
last_week_reshaped = last_week_scaled.reshape((1, 1, last_week_scaled.shape[1]))

next_week_pred_proba = model.predict(last_week_reshaped)
next_week_pred = (next_week_pred_proba > 0.5).astype(int)

st.header("Predicción para los proximos 5 dias")
st.write(f"Tendencia Predicha: {'Alcista (Sube)' if next_week_pred == 1 else 'Bajista (Baja)'}")
st.write(f"Probabilidad: {next_week_pred_proba[0][0]*100:.2f}%")

# Calcular la media móvil de 20 días
df['Moving_Average'] = df['Close'].rolling(window=20).mean()
df.dropna()

# Gráfica de precios de cierre y media móvil
st.subheader("Gráfica de Precios de Cierre y Media Móvil")
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Precio de Cierre')
plt.plot(df['Date'], df['Moving_Average'], label='Media Móvil (20 días)')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.title(f'Precio de Cierre y Media Móvil de {option}')
plt.legend()
plt.grid(True)
st.pyplot(plt)
 

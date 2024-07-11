
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix
from alpha_vantage.timeseries import TimeSeries

from datetime import datetime
import yfinance as yf

plt.ion()

def save_dataset(symbol):
    data = yf.download(symbol)

    return data 

df = save_dataset('MSFT')

df = df.sort_values('Date')
df.reset_index(inplace=True)

# DATA CLEANING AND preprocessing

# Calcular retornos semanales
df['Weekly_Return'] = df['Close'].pct_change(5)  

# Calculate the weekly trend (1 for positive, 0 for negative or if equals 0)
df['Weekly_Trend'] = np.where(df['Weekly_Return'] > 0, 1, 0)

df = df.dropna()

# Create weekly features (log returns of the last 4 weeks)
for i in range(1, 5):
    df[f'Week_Return_{i}'] = df['Weekly_Return'].shift(i)

df = df.dropna()

train_size = int(len(df) * 0.7)
train_data = df[:train_size]
test_data = df[train_size:]

features = ['Week_Return_1', 'Week_Return_2', 'Week_Return_3', 'Week_Return_4']
X_train = train_data[features]
y_train = train_data['Weekly_Trend']
X_test = test_data[features]
y_test = test_data['Weekly_Trend']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape para LSTM (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train_reshaped.shape[2]), kernel_regularizer=l2(0.0005), return_sequences=True),
    Dropout(0.02),
    LSTM(50, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.02),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005))
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

history = model.fit(X_train_reshaped, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=1)

y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int)

model.save('trendaizer.h5')

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Compare with naive prediction
naive_pred = np.ones(len(y_test)) * y_train.mode()[0]
naive_accuracy = accuracy_score(y_test, naive_pred)
print(f"Naive Accuracy: {naive_accuracy}")

plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_dates = test_data['Date'].values

comparison_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_Trend': y_test,
    'Predicted_Trend': y_pred.flatten(),
    'Predicted_Probability': y_pred_proba.flatten()
})

comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
comparison_df = comparison_df[comparison_df['Date'].dt.weekday == 4]  # 4 is the index for friday
comparison_df = comparison_df.sort_values('Date')

print(comparison_df)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
stock_symbol = "AAPL"  # Change this to the symbol of the company you want
start_date = "2010-01-01"
end_date = "2021-09-01"
data = yf.download(stock_symbol, start=start_date, end=end_date)
data = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append((sequence, target))
    return np.array(sequences)
seq_length = 10  # You can adjust this parameter
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_sequences[:, 0], train_sequences[:, 1], epochs=50, batch_size=64)
predicted_stock_prices = []
for i in range(len(test_sequences)):
    input_seq = test_sequences[i, 0].reshape(1, seq_length, 1)
    predicted_price = model.predict(input_seq)[0, 0]
    predicted_stock_prices.append(predicted_price)
predicted_stock_prices = scaler.inverse_transform(np.array(predicted_stock_prices).reshape(-1, 1))
plt.figure(figsize=(12, 6))
plt.plot(data[train_size + seq_length:], label='Actual Stock Prices')
plt.plot(predicted_stock_prices, label='Predicted Stock Prices')
plt.legend()
plt.show()

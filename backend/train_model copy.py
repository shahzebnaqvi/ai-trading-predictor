import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_model(symbol='AAPL'):
    # Create models directory
    os.makedirs('data/models', exist_ok=True)
    
    # Get data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='60d', interval='5m')
    
    # Add indicators (same as will be used in prediction)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
    
    # Select features (must match prediction)
    features = df[['Close', 'Volume', 'SMA_20', 'RSI']].dropna()
    
    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    seq_length = 10
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i-seq_length:i])
        y.append(scaled[i, 0])  # Predict Close price
    
    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 4)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile with explicit loss
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Train and save
    model.fit(np.array(X), np.array(y), epochs=10, batch_size=32)
    model.save(f'data/models/{symbol}_model.h5')
    joblib.dump(scaler, f'data/models/{symbol}_scaler.pkl')
    print(f"Model saved for {symbol}")

if __name__ == '__main__':
    train_model('AAPL')
    # Uncomment to train additional models
    # train_model('MSFT')
    # train_model('TSLA')

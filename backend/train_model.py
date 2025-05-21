import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import logging
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polygon.io configuration
POLYGON_API_KEY = "kHUK1CZ0BH2W5HpZIz8eLk_W9sXrp9Zi"  # Replace with your actual key
BASE_URL = "https://api.polygon.io"

def get_polygon_data(symbol, days=180, interval="minute"):
    """Fetch data from Polygon.io API"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/{interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("resultsCount", 0) == 0:
            raise ValueError(f"No results for {symbol}")
        
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        
        # Handle missing values
        df = df.fillna(method="ffill").dropna()
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df[["Open", "High", "Low", "Close", "Volume"]]
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return None

def prepare_features(df):
    """Create technical indicators"""
    try:
        features = pd.DataFrame({
            "Close": df["Close"],
            "Volume": df["Volume"],
            "SMA_20": df["Close"].rolling(20).mean(),
            "EMA_12": df["Close"].ewm(span=12).mean(),
            "EMA_26": df["Close"].ewm(span=26).mean()
        })
        
        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        features["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # MACD
        features["MACD"] = features["EMA_12"] - features["EMA_26"]
        
        # Bollinger Bands
        features["Upper_BB"] = features["SMA_20"] + (2 * df["Close"].rolling(20).std())
        features["Lower_BB"] = features["SMA_20"] - (2 * df["Close"].rolling(20).std())
        
        # Handle outliers (clip extreme values)
        for col in ["Close", "Volume", "MACD"]:
            features[col] = features[col].clip(lower=features[col].quantile(0.01), upper=features[col].quantile(0.99))
        
        return features.dropna()
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

def create_sequences(data, seq_length=20):
    """Create time-series sequences"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict Close price
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Simplified LSTM model with regularization"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        LSTM(32, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

def train_model(symbol, days=180, interval="minute", seq_length=20):
    """Complete training pipeline"""
    try:
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/plots", exist_ok=True)
        
        # Get data
        df = get_polygon_data(symbol, days, interval)
        if df is None:
            return False, None
            
        # Prepare features
        features = prepare_features(df)
        logger.info(f"Features shape: {features.shape}")
        
        # Scale features individually
        scalers = {}
        scaled = np.zeros_like(features)
        for i, col in enumerate(features.columns):
            if col == "RSI":
                scalers[col] = StandardScaler()
            else:
                scalers[col] = MinMaxScaler()
            scaled[:, i] = scalers[col].fit_transform(features[[col]].values).ravel()
        
        # Train-test split (temporal)
        train_size = int(0.7 * len(scaled))
        val_size = int(0.2 * len(scaled))
        train_data = scaled[:train_size]
        val_data = scaled[train_size:train_size + val_size]
        test_data = scaled[train_size + val_size:]
        
        # Create sequences
        X_train, y_train = create_sequences(train_data, seq_length)
        X_val, y_val = create_sequences(val_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)
        
        # Build and train model
        model = build_model((seq_length, features.shape[1]))
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model assets
        model.save(f"data/models/{symbol}_model.h5")
        joblib.dump(scalers, f"data/models/{symbol}_scalers.pkl")
        joblib.dump(features.columns, f"data/models/{symbol}_features.pkl")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"Training History for {symbol}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"data/plots/{symbol}_loss.png")
        plt.close()
        
        logger.info(f"Successfully trained model for {symbol}")
        return True, (model, scalers, features.columns, X_test, y_test, history)
        
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {str(e)}")
        return False, None

def predict(model, scalers, features_columns, X_test, y_test, symbol):
    """Make predictions and evaluate"""
    try:
        # Predict
        y_pred = model.predict(X_test)
        
        # Inverse scale predictions and actual values
        close_scaler = scalers["Close"]
        y_pred = close_scaler.inverse_transform(y_pred)
        y_test = close_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mae = np.mean(np.abs(y_pred - y_test))
        directional_accuracy = np.mean((np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_test[1:] - y_test[:-1])).astype(int))
        
        logger.info(f"{symbol} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Directional Accuracy: {directional_accuracy:.4f}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"Predictions vs Actual for {symbol}")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.legend()
        plt.savefig(f"data/plots/{symbol}_predictions.png")
        plt.close()
        
        return y_pred, y_test
        
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {str(e)}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model for a given symbol.")
    parser.add_argument("symbol", type=str, help="Ticker symbol (e.g., AAPL, X:BTCUSD)")
    parser.add_argument("--days", type=int, default=180, help="Number of days of data to use")
    parser.add_argument("--interval", type=str, default="minute", help="Data interval (minute, hour, day)")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length for LSTM")

    args = parser.parse_args()

    logger.info(f"Training model for {args.symbol}...")
    success, train_result = train_model(args.symbol, days=args.days, interval=args.interval, seq_length=args.seq_length)

    if success:
        model, scalers, features_columns, X_test, y_test, history = train_result
        logger.info(f"Evaluating model for {args.symbol}...")
        y_pred, y_test = predict(model, scalers, features_columns, X_test, y_test, args.symbol)
        logger.info(f"Model training and evaluation completed for {args.symbol}")
    else:
        logger.error(f"Failed to train model for {args.symbol}")
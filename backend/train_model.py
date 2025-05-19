import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polygon.io configuration
POLYGON_API_KEY = "kHUK1CZ0BH2W5HpZIz8eLk_W9sXrp9Zi"  # Replace with your actual key
BASE_URL = "https://api.polygon.io"

def get_polygon_data(symbol, days=60, interval="minute"):
    """Fetch data from Polygon.io API"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/{interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("resultsCount", 0) == 0:
            raise ValueError(f"No results for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume"
        }, inplace=True)
        
        return df[["Open", "High", "Low", "Close", "Volume"]]
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Polygon: {str(e)}")
        return None

def prepare_features(df):
    """Create technical indicators"""
    try:
        # Price features
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
    """Enhanced LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae"])
    return model

def train_model(symbol, days=60, interval="minute"):
    """Complete training pipeline with Polygon.io data"""
    try:
        os.makedirs("data/models", exist_ok=True)
        
        # Get data from Polygon
        df = get_polygon_data(symbol, days, interval)
        if df is None:
            return False
            
        # Prepare features
        features = prepare_features(df)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
        
        # Create sequences
        seq_length = 20
        X, y = create_sequences(scaled, seq_length)
        
        # Build and train model
        model = build_model((seq_length, features.shape[1]))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=5)
        ]
        
        history = model.fit(
            X, y,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model assets
        model.save(f"data/models/{symbol}_model.h5")
        joblib.dump(scaler, f"data/models/{symbol}_scaler.pkl")
        joblib.dump(features.columns, f"data/models/{symbol}_features.pkl")
        
        logger.info(f"Successfully trained model for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {str(e)}")
        return False

if __name__ == "__main__":
    # # Assets to train (using Polygon.io compatible symbols)
    # assets = {
    #     "US Stocks": ["AAPL", "MSFT", "TSLA"],
    #     "Forex": ["C:EURUSD", "C:GBPUSD", "C:USDJPY"],
    #     "Crypto": ["X:BTCUSD", "X:ETHUSD"]
    # }
    
    # # Train models
    # for asset_type, symbols in assets.items():
    #     logger.info(f"\nTraining {asset_type} models...")
    #     for symbol in symbols:
    #         logger.info(f"\nTraining model for {symbol}")
    #         success = train_model(symbol)
            
    #         if not success:
    #             logger.warning(f"Skipping {symbol} due to errors")
    #             continue
    parser = argparse.ArgumentParser(description="Train model for a given symbol.")
    parser.add_argument("symbol", type=str, help="Ticker symbol (e.g., AAPL, X:BTCUSD)")
    parser.add_argument("--days", type=int, default=60, help="Number of days of data to use")
    parser.add_argument("--interval", type=str, default="minute", help="Data interval (minute, hour, day)")

    args = parser.parse_args()

    logger.info(f"Training model for {args.symbol}...")
    success = train_model(args.symbol, days=args.days, interval=args.interval)

    if success:
        logger.info(f"Model trained successfully for {args.symbol}")
    else:
        logger.error(f"Failed to train model for {args.symbol}")
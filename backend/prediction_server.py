import os
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import logging
from datetime import datetime, timedelta

# Set Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Polygon.io configuration
POLYGON_API_KEY = "kHUK1CZ0BH2W5HpZIz8eLk_W9sXrp9Zi"  # Replace with your actual key
BASE_URL = "https://api.polygon.io"

def get_polygon_data(symbol, days=7, interval="minute", min_rows=20):
    """Fetch data from Polygon.io with fallback for insufficient data"""
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
        df = df.ffill().dropna()
        
        # Fallback: Increase days if insufficient data (max 14 days)
        if len(df) < min_rows:
            new_days = min(days + 7, 14)
            logger.warning(f"Only {len(df)} rows for {symbol}, retrying with {new_days} days")
            if new_days == days:
                raise ValueError(f"Cannot fetch enough data for {symbol}")
            return get_polygon_data(symbol, days=new_days, interval=interval, min_rows=min_rows)
        
        logger.info(f"Fetched {len(df)} rows for {symbol} with interval {interval}")
        logger.info(f"Data stats: Close mean={df['Close'].mean():.2f}, std={df['Close'].std():.2f}")
        return df[["Open", "High", "Low", "Close", "Volume"]]
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Polygon: {str(e)}")
        return None

def prepare_features(df, seq_length=20, is_update=False):
    """Feature preparation with minimal data loss, handling small DataFrames"""
    try:
        # Adjust min_periods for small DataFrames and specific windows
        min_periods = 1 if is_update else min(20, len(df))
        rsi_min_periods = min(14, min_periods)  # RSI window is 14
        
        features = pd.DataFrame({
            "Close": df["Close"],
            "Volume": df["Volume"],
            "SMA_20": df["Close"].rolling(20, min_periods=min_periods).mean(),
            "EMA_12": df["Close"].ewm(span=12, adjust=False, min_periods=min_periods).mean(),
            "EMA_26": df["Close"].ewm(span=26, adjust=False, min_periods=min_periods).mean(),
        })
        
        # RSI with specific min_periods
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=rsi_min_periods).mean()
        avg_loss = loss.rolling(14, min_periods=rsi_min_periods).mean()
        features["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # MACD
        features["MACD"] = features["EMA_12"] - features["EMA_26"]
        
        # Bollinger Bands
        std = df["Close"].rolling(20, min_periods=min_periods).std()
        features["Upper_BB"] = features["SMA_20"] + (2 * std)
        features["Lower_BB"] = features["SMA_20"] - (2 * std)
        
        # Handle outliers
        for col in ["Close", "Volume", "MACD"]:
            features[col] = features[col].clip(lower=features[col].quantile(0.01), upper=features[col].quantile(0.99))
        
        # Ensure enough rows
        features = features.ffill().dropna()
        logger.info(f"Features shape: {features.shape}, is_update={is_update}")
        if len(features) < (seq_length if not is_update else 1):
            logger.warning(f"Features reduced to {len(features)} rows after processing")
        logger.debug(f"Feature sample: {features.iloc[-1].to_dict()}")
        return features
    except Exception as e:
        logger.error(f"Feature preparation failed: {str(e)}")
        raise

def inverse_transform_prediction(scalers, features, prediction, feature_name="Close"):
    """Inverse transform predictions using the correct scaler"""
    feature_index = list(features.columns).index(feature_name)
    dummy = np.zeros((1, features.shape[1]))
    dummy[0, feature_index] = prediction
    transformed = scalers[feature_name].inverse_transform(dummy)[0, feature_index]
    logger.info(f"Raw prediction: {prediction}, Inverse transformed: {transformed}")
    return transformed

def plot_predictions(symbol, actual, predicted, timestamp):
    """Plot actual vs predicted prices for debugging"""
    try:
        os.makedirs("data/plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(actual, label="Actual")
        plt.plot(predicted, label="Predicted")
        plt.title(f"Predictions for {symbol}")
        plt.xlabel("Time Step")
        plt.ylabel("Close Price")
        plt.legend()
        plot_path = f"data/plots/{symbol}_pred_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved prediction plot for {symbol} at {plot_path}")
    except Exception as e:
        logger.error(f"Failed to plot predictions for {symbol}: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        seq_length = data.get('seq_length', 20)  # Match training default
        
        # Check for model assets
        model_path = f'data/models/{symbol}_model.h5'
        scalers_path = f'data/models/{symbol}_scalers.pkl'
        features_path = f'data/models/{symbol}_features.pkl'
        
        if not all(os.path.exists(p) for p in [model_path, scalers_path, features_path]):
            return jsonify({
                'status': 'error',
                'error': f'Model assets missing for {symbol}'
            })
        
        # Load model assets
        custom_objects = {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError()
        }
        trained_features = joblib.load(features_path)
        scalers = joblib.load(scalers_path)
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Log scaler ranges for Close
        close_scaler = scalers['Close']
        if hasattr(close_scaler, 'data_min_') and hasattr(close_scaler, 'data_max_'):
            logger.info(f"Close scaler range: min={close_scaler.data_min_[0]:.2f}, max={close_scaler.data_max_[0]:.2f}")
        
        # Get 1-minute data
        minute_data = get_polygon_data(symbol, days=7, interval="minute", min_rows=seq_length + 20)
        if minute_data is None or len(minute_data) < seq_length:
            return jsonify({
                'status': 'error',
                'error': f'Insufficient historical data (need at least {seq_length} periods after retries)'
            })
        
        # Prepare features
        features = prepare_features(minute_data, seq_length, is_update=False)
        if len(features) < seq_length:
            return jsonify({
                'status': 'error',
                'error': f'Insufficient feature data (only {len(features)} rows after processing)'
            })
        
        # Verify feature alignment
        if not all(f in features.columns for f in trained_features):
            missing = set(trained_features) - set(features.columns)
            return jsonify({
                'status': 'error',
                'error': f'Missing trained features: {missing}'
            })
        
        # Ensure correct feature order
        features = features[trained_features]
        current_price = float(minute_data['Close'].iloc[-1])
        logger.info(f"Current price for {symbol}: {current_price:.2f}")
        predictions = []
        actual_prices = minute_data['Close'].iloc[-seq_length:].values
        
        # Predict next 5 minutes
        latest_features = features.copy()
        for minute in range(1, 6):
            # Scale features
            scaled = np.zeros_like(latest_features.iloc[-seq_length:])
            for i, col in enumerate(trained_features):
                scaled[:, i] = scalers[col].transform(latest_features[[col]].iloc[-seq_length:].values).ravel()
            
            logger.info(f"Scaled input for {symbol} minute {minute}: {scaled[-1]}")
            X = scaled.reshape(1, seq_length, len(trained_features))
            scaled_pred = model.predict(X, verbose=0)[0][0]
            
            # Inverse transform prediction
            pred_price = inverse_transform_prediction(scalers, features, scaled_pred, "Close")
            
            # Validate prediction
            max_change = current_price * 0.1  # Relaxed to 10% for TSLA
            if abs(pred_price - current_price) > max_change:
                logger.warning(f"Unrealistic prediction for {symbol} minute {minute}: {pred_price}, capping to {current_price + np.sign(pred_price - current_price) * max_change}")
                pred_price = current_price + np.sign(pred_price - current_price) * max_change
            
            predictions.append({
                'minute': minute,
                'price': round(float(pred_price), 2),
                'change': round(float(pred_price - current_price), 2),
                'change_pct': round(((pred_price - current_price) / current_price) * 100, 2)
            })
            
            # Update features
            new_row = pd.DataFrame({
                'Open': pred_price,
                'High': pred_price,
                'Low': pred_price,
                'Close': pred_price,
                'Volume': minute_data['Volume'].iloc[-1]
            }, index=[minute_data.index[-1] + timedelta(minutes=1)])
            temp_df = pd.concat([minute_data.iloc[-(seq_length + 20):], new_row])
            new_features = prepare_features(temp_df, seq_length, is_update=True)
            if len(new_features) < 1:
                logger.error(f"Failed to generate new features for {symbol} minute {minute}")
                break
            logger.info(f"New features for {symbol} minute {minute}: {new_features.iloc[-1].to_dict()}")
            latest_features = pd.concat([latest_features.iloc[1:], new_features.iloc[-1:]])

        # Get 5-minute prediction with retry
        five_min_data = None
        for attempt in range(2):
            try:
                five_min_data = get_polygon_data(symbol, days=7 + attempt * 7, interval="5minute", min_rows=seq_length)
                break
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} for 5-minute data: {str(e)}")
        
        if five_min_data is None or len(five_min_data) < seq_length:
            logger.warning(f"Aggregating 1-minute data for 5-minute prediction due to fetch failure")
            five_min_data = minute_data.resample('5min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).ffill().dropna()
        
        if len(five_min_data) >= seq_length:
            five_min_features = prepare_features(five_min_data, seq_length, is_update=False)[trained_features]
            scaled_5m = np.zeros_like(five_min_features.iloc[-seq_length:])
            for i, col in enumerate(trained_features):
                input_data = five_min_features[[col]].iloc[-seq_length:].values
                scaled_5m[:, i] = scalers[col].transform(input_data).ravel()
                logger.debug(f"Scaler input for {col}: min={input_data.min():.2f}, max={input_data.max():.2f}")
            logger.info(f"Scaled 5-minute input for {symbol}: {scaled_5m[-1]}")
            X_5m = scaled_5m.reshape(1, seq_length, len(trained_features))
            scaled_5m_pred = model.predict(X_5m, verbose=0)[0][0]
            five_min_pred = inverse_transform_prediction(scalers, five_min_features, scaled_5m_pred, "Close")
            
            # Validate
            max_change_5m = current_price * 0.15  # Relaxed to 15% for TSLA
            if abs(five_min_pred - current_price) > max_change_5m:
                logger.warning(f"Unrealistic 5-minute prediction for {symbol}: {five_min_pred}, capping to {current_price + np.sign(five_min_pred - current_price) * max_change_5m}")
                five_min_pred = current_price + np.sign(five_min_pred - current_price) * max_change_5m
        else:
            logger.warning(f"Using last 1-minute prediction for 5-minute due to insufficient data")
            five_min_pred = predictions[-1]['price'] if predictions else current_price
        
        # Plot predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predicted_prices = [p['price'] for p in predictions]
        if len(predicted_prices) > 0:
            plot_predictions(symbol, actual_prices[:len(predicted_prices)], predicted_prices, timestamp)
        
        logger.info(f"Predictions for {symbol}: {predictions}")
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'minute_predictions': predictions,
            'five_min_prediction': round(float(five_min_pred), 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    logger.info("Starting prediction server...")
    serve(app, host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from waitress import serve
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def get_historical_data(symbol, interval='1m', days=1):
    """Get historical data with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f'{days}d', interval=interval)
        return hist.dropna() if not hist.empty else None
    except Exception as e:
        logger.error(f"Data fetch failed for {symbol}: {str(e)}")
        return None

def prepare_features(df):
    """Feature preparation that exactly matches training"""
    try:
        # Calculate all features used during training
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # MACD calculation
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        # Bollinger Bands
        std = df['Close'].rolling(20).std()
        df['Upper_BB'] = df['SMA_20'] + (2 * std)
        df['Lower_BB'] = df['SMA_20'] - (2 * std)
        
        return df[['Close', 'Volume', 'EMA_12', 'EMA_26', 'SMA_20', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB']]
    except Exception as e:
        logger.error(f"Feature preparation failed: {str(e)}")
        raise

def inverse_transform_prediction(scaler, features, prediction, feature_index=0):
    """Properly inverse transform predictions"""
    # Create dummy array with same shape as original features
    dummy = np.zeros((1, features.shape[1]))
    dummy[0, feature_index] = prediction
    
    # Inverse transform
    inverted = scaler.inverse_transform(dummy)
    return inverted[0, feature_index]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        
        # Load model assets
        trained_features = joblib.load(f'data/models/{symbol}_features.pkl')
        model = load_model(f'data/models/{symbol}_model.h5')
        scaler = joblib.load(f'data/models/{symbol}_scaler.pkl')

        
        # Get data
        minute_data = get_historical_data(symbol, '1m')
        if minute_data is None or len(minute_data) < 20:
            return jsonify({
                'status': 'error', 
                'error': 'Insufficient historical data (need at least 20 periods)'
            })
        
        # Prepare features with exact match to training
        features = prepare_features(minute_data)
        
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
        predictions = []
        
        # Predict next 5 minutes
        for minute in range(1, 6):
            # Scale and predict
            scaled = scaler.transform(features.iloc[-10:])
            X = scaled.reshape(1, 10, len(trained_features))
            scaled_pred = model.predict(X)[0][0]
            
            # Proper inverse transform
            pred_price = inverse_transform_prediction(scaler, features.iloc[-10:], scaled_pred)
            
            # Validate prediction is reasonable
            if abs(pred_price - current_price) > current_price * 0.05:  # Max 5% change
                pred_price = current_price * (1 + np.sign(scaled_pred) * 0.01)  # Cap at 1% change
            
            predictions.append({
                'minute': minute,
                'price': round(float(pred_price), 2),
                'change': round(float(pred_price - current_price), 2),
                'change_pct': round(((pred_price - current_price) / current_price) * 100, 2)
            })
            
            # Update features with properly scaled prediction
            new_row = features.iloc[-1:].copy()
            new_row['Close'] = pred_price
            features = pd.concat([features.iloc[1:], new_row])
        
        # Get 5-minute aggregated prediction
        five_min_data = get_historical_data(symbol, '5m', days=2)
        if five_min_data is not None and len(five_min_data) >= 5:
            five_min_features = prepare_features(five_min_data)[trained_features]
            scaled_5m = scaler.transform(five_min_features.iloc[-5:])
            X_5m = scaled_5m.reshape(1, 5, len(trained_features))
            scaled_5m_pred = model.predict(X_5m)[0][0]
            five_min_pred = inverse_transform_prediction(scaler, five_min_features.iloc[-5:], scaled_5m_pred)
            
            # Validate 5-minute prediction
            if abs(five_min_pred - current_price) > current_price * 0.1:  # Max 10% change
                five_min_pred = current_price * (1 + np.sign(scaled_5m_pred) * 0.05)  # Cap at 5% change
        else:
            five_min_pred = predictions[-1]['price']  # Fallback
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'minute_predictions': predictions,
            'five_min_prediction': round(float(five_min_pred), 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    logger.info("Starting prediction server with proper scaling...")
    serve(app, host="0.0.0.0", port=5000)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from main import load_and_engineer_data, build_and_train_models, get_sentiment

app = FastAPI()

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and data
MODEL_CACHE = {}

def get_models():
    if "data" not in MODEL_CACHE:
        print("Training models...")
        data = load_and_engineer_data()
        models = build_and_train_models(data)
        MODEL_CACHE["data"] = data
        MODEL_CACHE["models"] = models
    return MODEL_CACHE["data"], MODEL_CACHE["models"]

@app.get("/api/market-status")
async def get_market_status():
    try:
        data, models = get_models()
        latest_data = data.iloc[-1]
        
        regimes = {0: 'Bull', 1: 'Bear', 2: 'Sideways'}
        
        return {
            "ticker": "^BSESN",
            "price": float(latest_data["Close"]),
            "regime": regimes.get(int(latest_data["Regime"]), "Unknown"),
            "rsi": float(latest_data["RSI"]),
            "volatility": float(latest_data["Annual_Volatility"]),
            "sharpe_ratio": float(latest_data["Sharpe_Ratio"]),
            "sentiment": float(latest_data["Average_Sentiment"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_regime(text: str):
    try:
        data, models = get_models()
        lstm_model, base_xgb, fusion_model, tokenizer, sentiment_model, scaler, price_features, macro_features, window = models
        
        # Sentiment
        live_sentiment_score = get_sentiment(text, tokenizer, sentiment_model)
        
        # LSTM input
        recent_price_data = data[price_features].iloc[-window:].copy()
        scaled_recent_price = scaler.transform(recent_price_data)
        lstm_input_flat = scaled_recent_price.reshape(1, -1)
        lstm_probs = lstm_model.predict_proba(lstm_input_flat)
        
        # Macro input
        recent_macro = data[macro_features].iloc[-1:].copy()
        xgb_probs = base_xgb.predict_proba(recent_macro)
        
        # Fusion
        sentiment_input = np.array([[live_sentiment_score]])
        meta_features = np.hstack((lstm_probs, xgb_probs, sentiment_input))
        fusion_probs = fusion_model.predict_proba(meta_features)[0]
        
        final_pred = int(np.argmax(fusion_probs))
        confidence = np.max(fusion_probs)
        
        regimes = {0: 'Bull Market', 1: 'Bear Market', 2: 'Sideways Market'}
        
        return {
            "prediction": regimes.get(final_pred, 'Unknown'),
            "confidence": float(confidence),
            "sentiment_score": float(live_sentiment_score),
            "probabilities": [float(p) for p in fusion_probs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

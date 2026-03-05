from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime
from .fetch import get_recent_data
from .predict import predict_next_24h, predict_next_7days

app = FastAPI(title="Bakkhali Weather Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Bakkhali Weather Prediction API",
        "endpoints": {
            "health": "/health",
            "predict_24h": "/api/predict/24h",
            "predict_7days": "/api/predict/7days"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/predict/24h")
async def predict_24h():
    try:
        print("Fetching recent data...")
        historical_df = get_recent_data(hours=240)
        
        print("Making predictions...")
        predictions = predict_next_24h(historical_df)
        
        return {
            "success": True,
            "predictions": predictions,
            "error": None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
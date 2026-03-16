from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime
from .fetch import get_recent_data
from .predict import predict_next_24h, predict_next_72h, predict_next_168h

app = FastAPI(title="Bakkhali Weather Prediction API", version="2.0.0")

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
            "predict_72h": "/api/predict/72h",
            "predict_168h": "/api/predict/168h",   
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/predict/24h")
async def predict_24h():
    """Predict next 24 hours — returns a flat list of hourly predictions."""
    try:
        print("Fetching recent data...")
        historical_df = get_recent_data(hours=240)

        print("Making 24h predictions...")
        predictions = predict_next_24h(historical_df)

        return {
            "success": True,
            "horizon": "24h",
            "predictions": predictions,
            "error": None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/72h")
async def predict_72h():
    """
    Predict the next 72 hours (3 days).

    Response shape:
    {
        "success": true,
        "horizon": "72h",
        "hourly": [ ...72 hourly dicts... ],
        "daily": [
            {
                "date": "2025-01-15",
                "day_label": "Tomorrow",
                "summary": {
                    "Temperature (°C)": {"min": 18.2, "max": 27.4, "avg": 22.1},
                    ...
                    "Total Precipitation (mm)": 3.4
                },
                "hourly": [ ...24 hourly dicts for this day... ]
            },
            ...  (3 day objects total)
        ]
    }
    """
    try:
        print("Fetching recent data...")
        historical_df = get_recent_data(hours=240)

        print("Making 72h predictions...")
        result = predict_next_72h(historical_df)

        return {
            "success": True,
            "horizon": "72h",
            "hourly": result["hourly"],
            "daily": result["daily"],
            "error": None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/predict/168h")
async def predict_168h():
    """
    Predict the next 168 hours (7 days).
    Same response shape as /api/predict/72h but with 7 day objects.
    """
    try:
        print("Fetching recent data...")
        historical_df = get_recent_data(hours=240)

        print("Making 168h predictions...")
        result = predict_next_168h(historical_df)

        return {
            "success": True,
            "horizon": "168h",
            "hourly": result["hourly"],
            "daily": result["daily"],
            "error": None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

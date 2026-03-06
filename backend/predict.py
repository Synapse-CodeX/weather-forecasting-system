import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_weather_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')
TARGETS_PATH = os.path.join(BASE_DIR, 'models', 'target_names.pkl')

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
target_names = joblib.load(TARGETS_PATH)

def convert_numpy(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

def engineer_features_exactly_like_notebook(df):
    
    
    df = df.copy()
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    
    df.set_index("datetime", inplace=True)
    
    if "Radiation (W/m²)" in df.columns:
        df["Radiation (W/m²)"] = df["Radiation (W/m²)"].interpolate(method="time")
    if "Cloud Coverage (%)" in df.columns:
        df["Cloud Coverage (%)"] = df["Cloud Coverage (%)"].interpolate(method="time")
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            
            df[col] = df[col].replace(-999, np.nan)
            
            df[col] = df.groupby(df.index.hour)[col].transform(
                lambda x: x.fillna(x.median())
            )
            
            df[col] = df[col].ffill().bfill()
    
    df.reset_index(inplace=True)
    

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    

    lag_features = [
        "Temperature (°C)",
        "Radiation (W/m²)",
        "Cloud Coverage (%)",
        "Precipitation (mm/hr)",
        "Humidity (%)",
        "Pressure (kPa)",
        "Wind Speed (m/s)"
    ]
    lags = [1, 2, 3, 24, 48, 72, 96, 120]
    
    for feature in lag_features:
        if feature in df.columns:
            for lag in lags:
                df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
    
    if "Radiation (W/m²)" in df.columns:
        df['rad_roll_mean_3'] = df['Radiation (W/m²)'].rolling(window=3, min_periods=1).mean()
        df['rad_roll_mean_6'] = df['Radiation (W/m²)'].rolling(window=6, min_periods=1).mean()
        df['rad_roll_std_6'] = df['Radiation (W/m²)'].rolling(window=6, min_periods=1).std()
    
    if "Wind Speed (m/s)" in df.columns:
        df['wind_roll_mean_3'] = df['Wind Speed (m/s)'].rolling(window=3, min_periods=1).mean()
        df['wind_roll_mean_6'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).mean()
        df['wind_roll_std_6'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).std()
    

    df.dropna(inplace=True)
    
    return df

def predict_next_24h(historical_df):
    
    print("Engineering features...")
    Z = engineer_features_exactly_like_notebook(historical_df)
    
   
    last_datetime = Z["datetime"].max()
    
    
    target_datetimes = pd.date_range(
        start=last_datetime + timedelta(hours=1),
        end=last_datetime + timedelta(hours=24),
        freq="H"
    )
    
    predictions = []
    
    for target_dt in target_datetimes:
        
        new_rows = pd.DataFrame({"datetime": [target_dt]})
        
        
        for col in Z.columns:
            if col not in new_rows.columns and col != 'datetime':
                new_rows[col] = None
        
        
        extended_df = pd.concat([Z, new_rows], ignore_index=True)
        extended_df["datetime"] = pd.to_datetime(extended_df["datetime"])
        
    
        extended_df["hour"] = extended_df["datetime"].dt.hour
        
        
        mask_new = extended_df["datetime"].isin([target_dt])
        
    
        last_10_days = Z[Z["datetime"] >= (last_datetime - timedelta(days=10))]
        
        for col in Z.columns:
            if col != 'datetime' and col in extended_df.columns:
                if pd.api.types.is_numeric_dtype(extended_df[col]):
                    hourly_means = last_10_days.groupby(last_10_days["datetime"].dt.hour)[col].mean()
                    
                    # Map to new rows
                    extended_df.loc[mask_new, col] = extended_df.loc[mask_new, "hour"].map(hourly_means)
        
    
        extended_df["hour"] = extended_df["datetime"].dt.hour
        extended_df["month"] = extended_df["datetime"].dt.month
        extended_df["year"] = extended_df["datetime"].dt.year
        
    
        extended_df["hour_sin"] = np.sin(2 * np.pi * extended_df["hour"] / 24)
        extended_df["hour_cos"] = np.cos(2 * np.pi * extended_df["hour"] / 24)
        extended_df["month_sin"] = np.sin(2 * np.pi * extended_df["month"] / 12)
        extended_df["month_cos"] = np.cos(2 * np.pi * extended_df["month"] / 12)
        
   
        final_input = extended_df.tail(1).drop(columns=['datetime'], errors='ignore')
        
      
        for feat in feature_names:
            if feat not in final_input.columns:
                final_input[feat] = 0
        
        
        final_input = final_input[feature_names]
        
       
        prediction = model.predict(final_input)[0]
        
       
        result = {}
        for i, name in enumerate(target_names):
            value = prediction[i]
            if isinstance(value, (np.floating, np.integer)):
                value = float(value)
            result[name] = round(value, 2)
        
        # Add metadata
        result['datetime'] = target_dt.strftime('%Y-%m-%d %H:00')
        result['hour'] = target_dt.hour
        
        predictions.append(result)
        
       
        new_row = result.copy()
        new_row['datetime'] = target_dt
        Z = pd.concat([Z, pd.DataFrame([new_row])], ignore_index=True)
    
    return convert_numpy(predictions)


if __name__ == "__main__":
    print("Model loaded successfully!")
    print(f"Targets: {target_names}")
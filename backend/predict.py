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



WEATHER_COLS = [
    "Temperature (°C)",
    "Radiation (W/m²)",
    "Cloud Coverage (%)",
    "Precipitation (mm/hr)",
    "Humidity (%)",
    "Pressure (kPa)",
    "Wind Speed (m/s)"
]


LONG_LAG_COLS = [
    "Radiation (W/m²)_lag96",
    "Radiation (W/m²)_lag120",
    "Temperature (°C)_lag96",
    "Temperature (°C)_lag120",
    "Humidity (%)_lag96",
    "Humidity (%)_lag120",
    "Wind Speed (m/s)_lag96",
    "Wind Speed (m/s)_lag120",
]


MONTHLY_PEAK_RADIATION = {
    1: 580, 2: 640, 3: 700, 4: 750, 5: 780, 6: 760,
    7: 630, 8: 620, 9: 670, 10: 660, 11: 600, 12: 555
}




def convert_numpy(obj):
    """Recursively convert numpy types to plain Python types for JSON serialisation."""
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




def theoretical_radiation(hour: int, month: int) -> float:
    """
    Approximate clear-sky solar radiation for Bakkhali (~21.5N latitude).
    Uses a Gaussian bell curve centred at solar noon (~12:30 local time).
    Returns W/m². Always 0 outside daylight hours (before 06:00, from 19:00).
    """
    if hour < 6 or hour >= 19:
        return 0.0
    peak = MONTHLY_PEAK_RADIATION.get(month, 680)
    solar_noon = 12.5
    sigma = 3.5
    raw = peak * np.exp(-0.5 * ((hour - solar_noon) / sigma) ** 2)
    return round(max(0.0, raw), 2)


def blend_radiation(predicted: float, hour: int, month: int,
                    day_offset: int) -> float:
    """
    Blend the model's predicted radiation with the theoretical clear-sky value.

    As day_offset grows the model's lags become increasingly contaminated by
    its own prior predictions, so we progressively trust the climatological
    curve more and the model less.

      day_offset 1  -> 85% model, 15% theoretical
      day_offset 3  -> 65% model, 35% theoretical
      day_offset 5  -> 45% model, 55% theoretical
      day_offset 7+ -> 25% model, 75% theoretical
    """
    if hour < 6 or hour >= 19:
        return 0.0
    theoretical = theoretical_radiation(hour, month)
    model_weight = max(0.25, 0.85 - (day_offset - 1) * 0.10)
    theo_weight = 1.0 - model_weight
    blended = (predicted * model_weight) + (theoretical * theo_weight)
    return round(max(0.0, blended), 2)


# ── Feature engineering ───────────────────────────────────────────────────────

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


# ── Core prediction loop ──────────────────────────────────────────────────────

def _predict_hours(historical_df, hours: int):
    """
    Core rolling autoregressive prediction loop.

    Predicts `hours` steps into the future one hour at a time. Each predicted
    hour is appended to the working dataframe Z and used as input for the next
    step (autoregressive / teacher-forcing-free inference).

    Drift mitigations applied inside the loop:

      Fix 2 — Long lag columns (lag96, lag120) are overridden with their last
               real observed values before every model.predict() call. This
               prevents those features from being filled entirely with
               compounding predicted values once we are >4 days out.

      Fix 3 — After prediction, radiation is blended with a physically-derived
               clear-sky curve (Gaussian centred on solar noon, scaled by month).
               The blend shifts progressively toward the theoretical curve the
               further the prediction horizon, counteracting the model's tendency
               to replicate Day 1 radiation patterns on Days 2-7.

    Returns a flat list of hourly prediction dicts.
    """
    print(f"Engineering features for {hours}h prediction...")
    Z = engineer_features_exactly_like_notebook(historical_df)

    
    real_long_lag_values = {}
    for lag_col in LONG_LAG_COLS:
        if lag_col in Z.columns:
            series = Z[lag_col].dropna()
            if not series.empty:
                real_long_lag_values[lag_col] = float(series.iloc[-1])

    last_datetime = Z["datetime"].max()
    today = datetime.now().date()

    target_datetimes = pd.date_range(
        start=last_datetime + timedelta(hours=1),
        periods=hours,
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
                    hourly_means = last_10_days.groupby(
                        last_10_days["datetime"].dt.hour
                    )[col].mean()
                    extended_df.loc[mask_new, col] = (
                        extended_df.loc[mask_new, "hour"].map(hourly_means)
                    )

   
        extended_df["hour"] = extended_df["datetime"].dt.hour
        extended_df["month"] = extended_df["datetime"].dt.month
        extended_df["year"] = extended_df["datetime"].dt.year
        extended_df["hour_sin"] = np.sin(2 * np.pi * extended_df["hour"] / 24)
        extended_df["hour_cos"] = np.cos(2 * np.pi * extended_df["hour"] / 24)
        extended_df["month_sin"] = np.sin(2 * np.pi * extended_df["month"] / 12)
        extended_df["month_cos"] = np.cos(2 * np.pi * extended_df["month"] / 12)

     
        final_input = extended_df.tail(1).drop(columns=['datetime'], errors='ignore').copy()

        for feat in feature_names:
            if feat not in final_input.columns:
                final_input[feat] = 0

        final_input = final_input[feature_names].copy()

    
        for lag_col, real_val in real_long_lag_values.items():
            if lag_col in final_input.columns:
                final_input[lag_col] = real_val

      
        prediction = model.predict(final_input)[0]

        result = {}
        for i, name in enumerate(target_names):
            value = prediction[i]
            if isinstance(value, (np.floating, np.integer)):
                value = float(value)
            result[name] = round(value, 2)

      
        hour = target_dt.hour
        month = target_dt.month
        day_offset = max(1, (target_dt.date() - today).days + 1)

    
        result["Radiation (W/m²)"] = blend_radiation(
            predicted=result.get("Radiation (W/m²)", 0.0),
            hour=hour,
            month=month,
            day_offset=day_offset
        )

     
        result["Humidity (%)"] = max(0.0, min(100.0, result.get("Humidity (%)", 0)))
        result["Cloud Coverage (%)"] = max(0.0, min(100.0, result.get("Cloud Coverage (%)", 0)))
        result["Precipitation (mm/hr)"] = max(0.0, result.get("Precipitation (mm/hr)", 0))
        result["Wind Speed (m/s)"] = max(0.0, result.get("Wind Speed (m/s)", 0))

  
        result['datetime'] = target_dt.strftime('%Y-%m-%d %H:00')
        result['hour'] = hour
        result['date'] = target_dt.strftime('%Y-%m-%d')
        result['day_label'] = _day_label(target_dt)
        result['day_offset'] = day_offset

        predictions.append(result)

       
        new_row = result.copy()
        new_row['datetime'] = target_dt
        for col in WEATHER_COLS:
            if col in result:
                new_row[col] = result[col]
        Z = pd.concat([Z, pd.DataFrame([new_row])], ignore_index=True)

    return convert_numpy(predictions)



def _day_label(dt: datetime) -> str:
    """Human-readable label: Today / Tomorrow / Day After Tomorrow / weekday date."""
    today = datetime.now().date()
    delta = (dt.date() - today).days
    if delta == 0:
        return "Today"
    elif delta == 1:
        return "Tomorrow"
    elif delta == 2:
        return "Day After Tomorrow"
    else:
        return dt.strftime("%A, %b %d")


def _group_by_day(flat_predictions: list) -> list:
    """
    Bucket a flat list of hourly predictions into per-day objects.
    Each object contains: date, day_label, summary (stats), hourly (full list).
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for p in flat_predictions:
        grouped[p['date']].append(p)

    result = []
    for date_str, hours in sorted(grouped.items()):
        result.append({
            "date": date_str,
            "day_label": hours[0]['day_label'],
            "summary": _daily_summary(hours),
            "hourly": hours
        })
    return result


def _daily_summary(hourly_predictions: list) -> dict:
    """Min / max / avg summary for each weather variable across a day's hours."""
    keys = [
        "Temperature (°C)",
        "Humidity (%)",
        "Precipitation (mm/hr)",
        "Wind Speed (m/s)",
        "Cloud Coverage (%)",
        "Pressure (kPa)",
        "Radiation (W/m²)"
    ]
    summary = {}
    for key in keys:
        values = [
            p[key] for p in hourly_predictions
            if key in p and p[key] is not None
        ]
        if values:
            summary[key] = {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "avg": round(sum(values) / len(values), 2)
            }

 
    precip = [p.get("Precipitation (mm/hr)", 0) for p in hourly_predictions]
    summary["Total Precipitation (mm)"] = round(sum(precip), 2)

    return summary



def predict_next_24h(historical_df):
    """Predict next 24 hours. Returns flat list of hourly dicts."""
    return _predict_hours(historical_df, hours=24)


def predict_next_72h(historical_df):
    """
    Predict next 72 hours (3 days).
    Returns {'hourly': [...72 dicts...], 'daily': [...3 day objects...]}.
    """
    flat = _predict_hours(historical_df, hours=72)
    return {"hourly": flat, "daily": _group_by_day(flat)}


def predict_next_168h(historical_df):
    """
    Predict next 168 hours (7 days).
    Returns {'hourly': [...168 dicts...], 'daily': [...7 day objects...]}.
    """
    flat = _predict_hours(historical_df, hours=168)
    return {"hourly": flat, "daily": _group_by_day(flat)}


if __name__ == "__main__":
    print("Model loaded successfully!")
    print(f"Targets: {target_names}")

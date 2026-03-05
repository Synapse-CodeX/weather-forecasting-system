import requests
import pandas as pd
from datetime import date, timedelta, datetime
import os

def fetch_historical_data(start_date="2020-01-01", end_date=None):
    
    if end_date is None:
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = yesterday
    
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=21.63&longitude=88.17"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m,"
        "shortwave_radiation,"
        "windspeed_10m,"
        "relative_humidity_2m,"
        "rain,"
        "cloudcover,"
        "surface_pressure"
        "&timezone=Asia/Kolkata"
    )
    
    response = requests.get(url)
    data = response.json()
    
    final_df = pd.DataFrame(data["hourly"])
    
    final_df = final_df.rename(columns={
        "time": "datetime",
        "temperature_2m": "Temperature (°C)",
        "shortwave_radiation": "Radiation (W/m²)",
        "windspeed_10m": "Wind Speed (m/s)",
        "relative_humidity_2m": "Humidity (%)",
        "rain": "Precipitation (mm/hr)",
        "cloudcover": "Cloud Coverage (%)",
        "surface_pressure": "Pressure (kPa)",
    })
    
    final_df['Pressure (kPa)'] = final_df['Pressure (kPa)'] / 10
    
    final_df["datetime"] = pd.to_datetime(final_df["datetime"])
    
    return final_df

def get_recent_data(hours=240):
    
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")  
    
    df = fetch_historical_data(start_date=start_date, end_date=end_date)
    
    return df.tail(hours + 120)
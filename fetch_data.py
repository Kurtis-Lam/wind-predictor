import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fetch_weather_data(latitude=22.3193, longitude=114.1694, days=365, save_csv=True, filename_tag=None):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    
    # GFS historical data is only available from 2020-07-01
    min_start_date = datetime(2020, 7, 1)
    if start_date < min_start_date:
        print(f"⚠️  GFS historical data only available from 2020-07-01. Adjusting start date.")
        start_date = min_start_date
        actual_days = (end_date - start_date).days
        print(f"    Fetching {actual_days} days instead of {days} days")
    else:
        actual_days = days
    
    print("=======================================================")
    print(f"⚠️  Fetching {actual_days} days of GFS historical data for ({latitude},{longitude})...")
    print(f"    Extracting weather parameters at multiple heights")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m", 
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "surface_pressure",
            "cloud_cover_low",
            "cloud_cover_mid",  
            "cloud_cover_high", 
            "wind_speed_10m",   
            "wind_direction_10m",  
            "wind_speed_100m", 
            "wind_direction_100m",
            "wind_gusts_10m"
        ],
        "timezone": "Asia/Hong_Kong",
    }

    try:
        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
        response.raise_for_status()
        data = response.json()
        print("✅ Data successfully retrieved from GFS Historical API")
        print("=======================================================")
    except Exception as e:
        print(f"API Error: {e}")
        if 'response' in locals() and response.status_code == 400:
            print(f"❌ Bad request parameters: {response.text}")
        return None

    hourly_data = data.get("hourly", {})
    if not hourly_data:
        print("❌ No hourly data found in API response")
        return None
        
    df = pd.DataFrame({
        "time": hourly_data["time"],
        "temperature_2m": hourly_data["temperature_2m"],
        "relative_humidity_2m": hourly_data["relative_humidity_2m"],
        "precipitation": hourly_data["precipitation"],
        "rain": hourly_data["rain"],
        "surface_pressure": hourly_data["surface_pressure"],
        "cloud_cover_low": hourly_data["cloud_cover_low"],
        "cloud_cover_mid": hourly_data["cloud_cover_mid"],
        "cloud_cover_high": hourly_data["cloud_cover_high"],
        "wind_speed_10m": hourly_data["wind_speed_10m"],    
        "wind_direction_10m": hourly_data["wind_direction_10m"],  
        "wind_speed_100m": hourly_data["wind_speed_100m"],
        "wind_direction_100m": hourly_data["wind_direction_100m"],
        "wind_gusts_10m": hourly_data["wind_gusts_10m"]
    })

    # Add location columns
    df["latitude"] = latitude
    df["longitude"] = longitude

    # Convert numeric columns to floats
    numeric_cols = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "rain", "surface_pressure",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "wind_speed_10m", "wind_direction_10m",
        "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize('Asia/Hong_Kong')

    print(f"⚠️  Missing values before cleaning: {df.isnull().sum().sum()}")
    df.interpolate(method='linear', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    print(f"⚠️  Missing values after cleaning: {df.isnull().sum().sum()}")
    print("=======================================================")

    # Calculate UV components for each height level
    for height in [10, 100]:
        speed_col = f"wind_speed_{height}m"
        dir_col = f"wind_direction_{height}m"
        
        df[f"u_wind_{height}m"] = -df[speed_col] * np.sin(np.deg2rad(df[dir_col]))
        df[f"v_wind_{height}m"] = -df[speed_col] * np.cos(np.deg2rad(df[dir_col]))

    if save_csv:
        os.makedirs("data", exist_ok=True)
        if filename_tag is None:
            filename_tag = f"{latitude}_{longitude}_{actual_days}"
        filename = f"data/{filename_tag}.txt"
        df.to_csv(filename, index=False)
        print(f"✅ Data saved to {filename}")
        print(f"   Total records: {len(df)}")
        print(f"   Date range: {df['time'].min().date()} to {df['time'].max().date()}")
        print(f"   Parameters: temperature, humidity, precipitation, rain, surface pressure, cloud cover (low/mid/high)")
        print(f"   Wind heights: 10m, 100m, including wind gusts")
        print("=======================================================")
    
    return df

def _clean_coord_str(s):
    return s.strip().replace(" ", "")

if __name__ == "__main__":
    print("=======================================================")

    # Latitude input
    lat_in = input("Enter latitude (default 22.3193): ").strip()
    if lat_in == "":
        latitude = 22.3193
        lat_tag = "22.3193"
    else:
        latitude = float(lat_in)
        lat_tag = _clean_coord_str(lat_in)

    # Longitude input
    lon_in = input("Enter longitude (default 114.1694): ").strip()
    if lon_in == "":
        longitude = 114.1694
        lon_tag = "114.1694"
    else:
        longitude = float(lon_in)
        lon_tag = _clean_coord_str(lon_in)

    yrs = int(input("Enter number of years to fetch (default 3): ") or 3)
    days = yrs * 365

    filename_tag = f"{lat_tag}_{lon_tag}_{days}"
    fetch_weather_data(latitude=latitude, longitude=longitude, days=days, filename_tag=filename_tag)
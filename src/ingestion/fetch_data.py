import requests
import pandas as pd

LAT, LON = 31.5497, 74.3436  # Lahore
TZ = "Asia/Karachi"


def fetch_weather(past_days=3):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                  "uv_index,wind_speed_10m,wind_direction_10m",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TZ,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])


def fetch_air_quality(past_days=3):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "us_aqi,pm2_5,pm10",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TZ,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])


if __name__ == "__main__":
    weather = fetch_weather()
    air = fetch_air_quality()
    df = weather.merge(air, on="time", how="outer")
    print(df.head())
    print(df.shape)
    print(df.isna().sum())
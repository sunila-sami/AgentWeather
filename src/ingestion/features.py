import numpy as np
import pandas as pd
from fetch_data import fetch_weather, fetch_air_quality
from impute import impute


def make_features(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Time features
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek

    # Wind direction as sin and cos
    angle = np.deg2rad(df["wind_direction_10m"])
    df["wind_dir_sin"] = np.sin(angle)
    df["wind_dir_cos"] = np.cos(angle)

    cols = ["temperature_2m", "relative_humidity_2m", "precipitation",
            "uv_index", "wind_speed_10m", "us_aqi"]

    for c in cols:
        # Change since the last hour
        df[f"{c}_change_1h"] = df[c].diff(1)
        # Average of the last 3 and 24 hours
        df[f"{c}_roll_3h"] = df[c].rolling(3, min_periods=1).mean()
        df[f"{c}_roll_24h"] = df[c].rolling(24, min_periods=1).mean()
        # Value 24 hours ago
        df[f"{c}_lag_24h"] = df[c].shift(24)

    return df


if __name__ == "__main__":
    weather = fetch_weather()
    air = fetch_air_quality()
    df = weather.merge(air, on="time", how="outer")

    clean = impute(df)
    features = make_features(clean)

    print(features.shape)
    print(features.columns.tolist())
    print(features.tail())
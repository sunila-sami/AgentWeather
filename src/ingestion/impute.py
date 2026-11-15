import numpy as np
import pandas as pd
from fetch_data import fetch_weather, fetch_air_quality


def impute(df):
    df = df.copy()

    # Convert the time column to real dates, sort it, remove duplicate hours
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")

    # Make sure every hour has a row (missing hours become empty rows)
    df = df.asfreq("h")

    # Count empty values before filling, so we can compare later
    missing_before = int(df.isna().sum().sum())

    # Rain: an empty value means no rain, so fill with 0
    if "precipitation" in df.columns:
        df["precipitation"] = df["precipitation"].fillna(0)

    # Layer 1: small gaps (up to 3 hours), estimate from the values around them
    df = df.interpolate(method="time", limit=3)

    # Layer 2: medium gaps (up to 6 hours), copy the last known value forward
    df = df.ffill(limit=6)

    # Layer 3: anything still empty, use the typical value for that hour of day,
    # and if that is also missing, use the overall median of the column
    hour = df.index.hour
    for col in df.columns:
        if df[col].isna().any():
            hour_median = df.groupby(hour)[col].transform("median")
            df[col] = df[col].fillna(hour_median)
            df[col] = df[col].fillna(df[col].median())

    # Count empty values after filling and show the result
    missing_after = int(df.isna().sum().sum())
    print(f"Missing values: {missing_before} -> {missing_after}")

    return df.reset_index()


if __name__ == "__main__":
    # Get weather and air quality data, then join them into one table
    weather = fetch_weather()
    air = fetch_air_quality()
    df = weather.merge(air, on="time", how="outer")

    # Test: remove some values on purpose to check that the filling works
    df.loc[10:11, "temperature_2m"] = np.nan   # 2-hour gap
    df.loc[30:35, "us_aqi"] = np.nan           # 6-hour gap

    # Fill the gaps and print the result
    clean = impute(df)
    print(clean.head())
    print(clean.isna().sum())
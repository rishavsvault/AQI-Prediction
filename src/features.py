import numpy as np
import pandas as pd

def create_features(df):

    df = df.copy()

    # Ensure datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values(["City", "Datetime"])

    # -----------------------------
    # Time Features
    # -----------------------------
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    df["Day"] = df["Datetime"].dt.day
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["DayOfYear"] = df["Datetime"].dt.dayofyear
    df["WeekOfYear"] = df["Datetime"].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)

    # -----------------------------
    # Grouped Feature Engineering
    # -----------------------------
    df_grouped = []

    for city, group in df.groupby("City"):

        group = group.copy()

        # Lag features
        for lag in [1, 7, 14, 30, 60, 90, 365]:
            group[f"AQI_lag_{lag}"] = group["AQI"].shift(lag)

        # Rolling stats
        for window in [7, 14, 30, 60]:
            shifted = group["AQI"].shift(1)
            group[f"AQI_roll_mean_{window}"] = shifted.rolling(window).mean()
            group[f"AQI_roll_std_{window}"] = shifted.rolling(window).std()
            group[f"AQI_roll_min_{window}"] = shifted.rolling(window).min()
            group[f"AQI_roll_max_{window}"] = shifted.rolling(window).max()

        # Momentum
        group["AQI_diff_1"] = group["AQI"] - group["AQI_lag_1"]
        group["AQI_diff_7"] = group["AQI"] - group["AQI_lag_7"]
        group["AQI_diff_30"] = group["AQI"] - group["AQI_lag_30"]

        # Expanding mean
        group["AQI_expanding_mean"] = (
            group["AQI"].shift(1).expanding().mean()
        )

        # Pollutants
        for pollutant in ["PM2.5", "PM10"]:
            for lag in [1, 7, 14, 30]:
                group[f"{pollutant}_lag_{lag}"] = group[pollutant].shift(lag)

            group[f"{pollutant}_roll_mean_7"] = (
                group[pollutant].shift(1).rolling(7).mean()
            )

        df_grouped.append(group)

    df = pd.concat(df_grouped)

    # Seasonal flag
    df["Winter"] = df["Month"].isin([11, 12, 1, 2]).astype(int)

    # -----------------------------
    # Final NaN handling
    # -----------------------------
    df = df.dropna()

    return df

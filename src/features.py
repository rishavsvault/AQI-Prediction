def create_features(df):

    for i in range(1, 8):
        df[f"lag_{i}"] = df["AQI"].shift(i)

    df["rolling_mean_7"] = df["AQI"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["AQI"].shift(1).rolling(7).std()

    df["month"] = df["Datetime"].dt.month
    df["day_of_week"] = df["Datetime"].dt.dayofweek

    df.dropna(inplace=True)
    return df

def create_features(df):
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    df["Day"] = df["Datetime"].dt.day
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek

    #  LAG FEATURES
    df["AQI_lag_1"] = df["AQI"].shift(1)
    df["AQI_lag_7"] = df["AQI"].shift(7)
    df["AQI_lag_30"] = df["AQI"].shift(30)

    #  ROLLING FEATURES
    df["AQI_roll_mean_7"] = df["AQI"].rolling(7).mean()
    df["AQI_roll_std_7"] = df["AQI"].rolling(7).std()

    df = df.dropna()

    return df


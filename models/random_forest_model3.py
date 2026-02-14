import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# ================= LOAD DATA =================
file_path = "data/Air_quality_data_interpolated.csv"
df = pd.read_csv(file_path)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Year"] = df["Datetime"].dt.year

# ================= SORT PROPERLY =================
df = df.sort_values(["City", "Datetime"])

# ================= BASE FEATURES =================
base_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

# ================= CREATE LAG FEATURES =================
lag_days = [1, 7, 30]

for lag in lag_days:
    df[f"AQI_lag_{lag}"] = df.groupby("City")["AQI"].shift(lag)
    
    for col in base_features:
        df[f"{col}_lag_{lag}"] = df.groupby("City")[col].shift(lag)

# ================= ROLLING FEATURES =================
df["AQI_roll7"] = df.groupby("City")["AQI"].rolling(7).mean().reset_index(level=0, drop=True)

for col in base_features:
    df[f"{col}_roll7"] = (
        df.groupby("City")[col]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

# ================= FINAL FEATURE LIST =================
lag_features = [col for col in df.columns if "lag" in col]
roll_features = [col for col in df.columns if "roll" in col]

features = base_features + lag_features + roll_features

# ================= DROP MISSING (AFTER LAGS) =================
df_clean = df.dropna(subset=features + ["AQI"])

# ================= TRAIN / TEST SPLIT (CHRONOLOGICAL) =================
last_year = df_clean["Year"].max()

train_df = df_clean[df_clean["Year"] < last_year]
test_df = df_clean[df_clean["Year"] == last_year]

X_train = train_df[features]
y_train = train_df["AQI"]

X_test = test_df[features]
y_test = test_df["AQI"]

print("Training Years:", sorted(train_df["Year"].unique()))
print("Testing Year:", last_year)

# ================= MODEL =================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ================= TIME SERIES CROSS VALIDATION =================
tscv = TimeSeriesSplit(n_splits=5)

cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_root_mean_squared_error"
)

print("\nTimeSeries CV RMSE:", -cv_scores.mean())

# ================= TRAIN PERFORMANCE =================
train_preds = rf.predict(X_train)

print("\n--- TRAIN PERFORMANCE ---")
print("Train R2:", r2_score(y_train, train_preds))
print("Train MAE:", mean_absolute_error(y_train, train_preds))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_preds)))

# ================= TEST PERFORMANCE =================
predictions = rf.predict(X_test)

print(f"\n--- TEST PERFORMANCE ({last_year}) ---")
print("Test R2:", r2_score(y_test, predictions))
print("Test MAE:", mean_absolute_error(y_test, predictions))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# ================= PLOT SAMPLE PREDICTIONS =================
plt.figure(figsize=(12, 6))

limit = 200
plt.plot(y_test.values[:limit], label="Actual AQI", alpha=0.8)
plt.plot(predictions[:limit], label="Predicted AQI", linestyle="--")

plt.title(f"AQI Prediction for {last_year} (First {limit} Samples)")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ================= FEATURE IMPORTANCE =================
importances = rf.feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_imp.head(20).plot(kind="bar")
plt.title("Top 20 Most Important Features")
plt.ylabel("Importance")
plt.grid(axis="y", alpha=0.3)
plt.show()

print("\nTop 10 Features Driving AQI:")
print(feature_imp.head(10))
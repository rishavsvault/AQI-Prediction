import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# ============================================================
# 1. LOAD DATA
# ============================================================
file_path = "data/Air_quality_data_interpolated.csv"
df = pd.read_csv(file_path)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Year"] = df["Datetime"].dt.year

# ============================================================
# 2. SORT BY CITY AND TIME (CRITICAL FOR LAGS)
# ============================================================
df = df.sort_values(["City", "Datetime"])

# ============================================================
# 3. DEFINE BASE POLLUTANT FEATURES
# ============================================================
base_features = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
    'NH3', 'CO', 'SO2', 'O3'
]

# ============================================================
# 4. CREATE LAG FEATURES (CITY-WISE)
# ============================================================
lag_days = [1, 7, 30]

for lag in lag_days:
    df[f"AQI_lag_{lag}"] = df.groupby("City")["AQI"].shift(lag)
    
    for col in base_features:
        df[f"{col}_lag_{lag}"] = df.groupby("City")[col].shift(lag)

# ============================================================
# 5. CREATE ROLLING FEATURES (CITY-WISE)
# ============================================================
df["AQI_roll7"] = (
    df.groupby("City")["AQI"]
    .rolling(window=7)
    .mean()
    .reset_index(level=0, drop=True)
)

for col in base_features:
    df[f"{col}_roll7"] = (
        df.groupby("City")[col]
        .rolling(window=7)
        .mean()
        .reset_index(level=0, drop=True)
    )

# ============================================================
# 6. FINAL FEATURE LIST
# ============================================================
lag_features = [col for col in df.columns if "_lag_" in col]
roll_features = [col for col in df.columns if "_roll7" in col]

features = base_features + lag_features + roll_features

# ============================================================
# 7. DROP NaNs CREATED BY LAGS
# ============================================================
df_clean = df.dropna(subset=features + ["AQI"])

# ============================================================
# 8. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
last_year = df_clean["Year"].max()

train_df = df_clean[df_clean["Year"] < last_year]
test_df = df_clean[df_clean["Year"] == last_year]

X_train = train_df[features]
y_train = train_df["AQI"]

X_test = test_df[features]
y_test = test_df["AQI"]

print("Training Years:", sorted(train_df["Year"].unique()))
print("Testing Year:", last_year)

# ============================================================
# 9. LINEAR REGRESSION PIPELINE WITH STANDARDIZATION
# ============================================================
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

# ============================================================
# 10. TIME SERIES CROSS VALIDATION
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)

cv_scores = cross_val_score(
    lr_pipeline,
    X_train,
    y_train,
    cv=tscv,
    scoring="neg_root_mean_squared_error"
)

print("\nTimeSeries CV RMSE:", -cv_scores.mean())

# ============================================================
# 11. TRAIN PERFORMANCE
# ============================================================
train_preds = lr_pipeline.predict(X_train)

print("\n--- LINEAR REGRESSION TRAIN PERFORMANCE ---")
print("Train R2:", r2_score(y_train, train_preds))
print("Train MAE:", mean_absolute_error(y_train, train_preds))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_preds)))

# ============================================================
# 12. TEST PERFORMANCE
# ============================================================
predictions = lr_pipeline.predict(X_test)

print(f"\n--- LINEAR REGRESSION TEST PERFORMANCE ({last_year}) ---")
print("Test R2:", r2_score(y_test, predictions))
print("Test MAE:", mean_absolute_error(y_test, predictions))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# ============================================================
# 13. PLOT SAMPLE PREDICTIONS
# ============================================================
plt.figure(figsize=(12, 6))

limit = 200
plt.plot(y_test.values[:limit], label="Actual AQI", alpha=0.8)
plt.plot(predictions[:limit], label="Predicted AQI", linestyle="--")

plt.title(f"Linear Regression AQI Prediction for {last_year} (First {limit} Samples)")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ============================================================
# 14. COEFFICIENT ANALYSIS
# ============================================================
model = lr_pipeline.named_steps["model"]
coefficients = pd.Series(model.coef_, index=features).sort_values(ascending=False)

print("\nTop 10 Positive Drivers of AQI:")
print(coefficients.head(10))

print("\nTop 10 Negative Drivers of AQI:")
print(coefficients.tail(10))
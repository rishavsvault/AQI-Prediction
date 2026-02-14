import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

file_path = "data/Air_quality_data.csv"
df = pd.read_csv(file_path)

df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Year'] = df['Datetime'].dt.year

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

df_clean = df.dropna(subset=features + ['AQI'])
last_year = df_clean['Year'].max()
train_df = df_clean[df_clean['Year'] < last_year]
test_df = df_clean[df_clean['Year'] == last_year]

X_train = train_df[features]
y_train = train_df['AQI']

X_test = test_df[features]
y_test = test_df['AQI']

print(f"Training on years: {train_df['Year'].unique()}")
print(f"Testing on year: {last_year}")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ================= CROSS VALIDATION =================
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

cv_rmse = -cv_scores.mean()
print(f"\nCV RMSE (5-fold): {cv_rmse:.4f}")
# ====================================================


# ================= TREE SANITY CHECK =================
print("\n--- Tree Info ---")
print("Number of trees:", len(rf.estimators_))
print("Depth of first tree:", rf.estimators_[0].get_depth())
# =====================================================

# ---- TRAIN PERFORMANCE CHECK ----
train_preds = rf.predict(X_train)

train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_r2 = r2_score(y_train, train_preds)

print("\n--- TRAIN Performance ---")
print(f"Train R2: {train_r2:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")

predictions = rf.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"\n--- Prediction Accuracy for {last_year} ---")
print(f"R^2 Score: {r2:.2f} (1.0 is perfect)")
print(f"Mean Absolute Error (MAE): {mae:.2f} (Average error in AQI points)")
print(f"Root Mean Sq Error (RMSE): {rmse:.2f}")


plt.figure(figsize=(12, 6))

limit = 100
plt.plot(range(limit), y_test.values[:limit], label='Actual AQI', color='blue', alpha=0.7)
plt.plot(range(limit), predictions[:limit], label='Predicted AQI', color='orange', linestyle='--')
plt.title(f"Actual vs Predicted AQI for Year {last_year} (First {limit} samples)")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

importances = rf.feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_imp.plot(kind='bar', color='teal')
plt.title("Which Pollutant Impacts AQI the Most?")
plt.ylabel("Importance")
plt.grid(axis='y', alpha=0.3)
plt.show()

print("\nTop Drivers of Pollution:")
print(feature_imp)

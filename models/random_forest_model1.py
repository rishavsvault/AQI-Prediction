import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================
# 1. Training the Full-Feature Model
# ==========================================
print("Training the ultimate AQI model...")

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load Data
df = pd.read_csv("data/Air_quality_data_interpolated.csv")

# Datetime + Season Feature
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Month'] = df['Datetime'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'PostMonsoon'

df['Season'] = df['Month'].apply(get_season)

# Features
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Season', 'City']
X_raw = df[features]
y = df['AQI']

# Encoding
X = pd.get_dummies(X_raw, columns=['Season', 'City'], drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 2. Evaluation & Metrics
# ==========================================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("-" * 40)
print("FINAL PERFORMANCE:")
print(f"R² Score:               {r2:.4f}")
print(f"Mean Squared Error:     {mse:.2f}")
print(f"Root Mean Squared Error:{rmse:.2f}")
print(f"Mean Absolute Error:    {mae:.2f}")
print("-" * 40)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "R2_Score": [r2],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae]
})

metrics_df.to_csv("outputs/model_metrics.csv", index=False)
print("Metrics saved to outputs/model_metrics.csv")

# ==========================================
# 3. Save Model & Graph
# ==========================================
joblib.dump(model, 'aqi_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("Model files saved successfully!")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, s=10)
plt.plot([0, 500], [0, 500], linestyle='--')
plt.xlabel("Actual AQI Values")
plt.ylabel("Predicted AQI")
plt.title(f"Actual vs Predicted (R²: {r2:.2%})")
plt.grid(True, alpha=0.3)
plt.savefig('outputs/aqi_prediction_graph.png', dpi=300)
print("Graph saved to outputs/aqi_prediction_graph.png")

# ==========================================
# 4. Prediction Function
# ==========================================
def predict_aqi(pm25, pm10, no, no2, nox, nh3, co, so2, o3, season, city):

    cols = joblib.load('model_columns.pkl')
    loaded_model = joblib.load('aqi_model.pkl')

    data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, season, city]],
                        columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                                 'CO', 'SO2', 'O3', 'Season', 'City'])

    encoded = pd.get_dummies(data, columns=['Season', 'City'], drop_first=True)
    encoded = encoded.reindex(columns=cols, fill_value=0)

    return loaded_model.predict(encoded)[0]


# ==========================================
# 5. Live Test
# ==========================================
print("\n--- TEST RUN ---")
test_val = predict_aqi(
    pm25=158, pm10=220, no=40, no2=80, nox=110,
    nh3=35, co=1.5, so2=15, o3=60,
    season='Winter', city='Delhi'
)

print("Input: Severe Winter Delhi scenario")
print(f"AI Prediction: {test_val:.2f}")

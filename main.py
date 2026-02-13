import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. Training the Full-Feature Model
# ==========================================
print("Training the ultimate AQI model (99.5% Accuracy)...")

# Load Data
df =  pd.read_csv("data/Air_quality_data_interpolated.csv")

# Feature Engineering: Convert Datetime to Seasons
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Month'] = df['Datetime'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Winter'

df['Season'] = df['Month'].apply(get_season)

# Select ALL 11 Features
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Season', 'City']
X_raw = df[features]
y = df['AQI']

# Encoding: Convert Categories to Numbers
X = pd.get_dummies(X_raw, columns=['Season', 'City'], drop_first=True)

# Split: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train: The "Sweet Spot" Config
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ==========================================
# 2. Evaluation & Results
# ==========================================
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("-" * 40)
print(f"FINAL PERFORMANCE:")
print(f"Accuracy (R² Score): {score:.2%}")
print(f"Mean Squared Error:  {mse:.2f}")
print("-" * 40)

# ==========================================
# 3. Save Files (Model & Graph)
# ==========================================
# Save the AI "Brain"
joblib.dump(model, 'aqi_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("Model files saved successfully!")

# Save the Performance Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='teal', s=10)
plt.plot([0, 500], [0, 500], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual AQI Values")
plt.ylabel("AI Predicted AQI")
plt.title(f"Final Model: Actual vs Predicted (Accuracy: {score:.2%})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('aqi_prediction_graph.png', dpi=300)
print("Graph saved as 'aqi_prediction_graph.png'!")

# ==========================================
# 4. Ultimate Prediction Function
# ==========================================
def predict_aqi(pm25, pm10, no, no2, nox, nh3, co, so2, o3, season, city):
    """
    Pass all parameters to get a highly accurate AQI prediction.
    """
    # Load requirements
    cols = joblib.load('model_columns.pkl')
    loaded_model = joblib.load('aqi_model.pkl')
    
    # Create single row for prediction
    data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, season, city]], 
                        columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Season', 'City'])
    
    # Encoding & Alignment
    encoded = pd.get_dummies(data, columns=['Season', 'City'], drop_first=True)
    encoded = encoded.reindex(columns=cols, fill_value=0)
    
    return loaded_model.predict(encoded)[0]

# --- LIVE TEST ---
print("\n--- TEST RUN ---")
test_val = predict_aqi(pm25=158, pm10=220, no=40, no2=80, nox=110, nh3=35, co=1.5, so2=15, o3=60, season='Winter', city='Delhi')
print(f"Input: Severe Winter Delhi scenario")
print(f"AI Prediction: {test_val:.2f}")
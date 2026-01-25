# ------------------------------------
# STEP 1: Import required libraries
# ------------------------------------
import pandas as pd
import numpy as np

# ------------------------------------
# STEP 2: Load the AQI dataset
# ------------------------------------
df = pd.read_csv("data/daily_aqi_raw.csv", parse_dates=["Date"])

# ------------------------------------
# STEP 3: Select only numeric (AQI) columns
# ------------------------------------
aqi_columns = df.select_dtypes(include=[np.number]).columns

# ------------------------------------
# STEP 4: Dictionary to store outliers
# ------------------------------------
outliers = {}

# ------------------------------------
# STEP 5: Apply IQR method city-wise
# ------------------------------------
for city in aqi_columns:
    
    # Remove missing values
    city_data = df[city].dropna()

    # Quartiles
    Q1 = city_data.quantile(0.25)
    Q3 = city_data.quantile(0.75)

    # Interquartile Range
    IQR = Q3 - Q1

    # IQR bounds (NO domain restriction)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    city_outliers = df[(df[city] < lower_bound) | (df[city] > upper_bound)]

    # Store results
    outliers[city] = city_outliers

    # Print results
    print(f"\n🔹 City: {city}")
    print(f"Q1 = {Q1:.2f}")
    print(f"Q3 = {Q3:.2f}")
    print(f"IQR = {IQR:.2f}")
    print(f"Lower Bound = {lower_bound:.2f}")
    print(f"Upper Bound = {upper_bound:.2f}")

    if city_outliers.empty:
        print("✅ No outliers detected")
    else:
        print("❗ Outliers detected:")
        print(city_outliers[['Date', city]])

# ------------------------------------
# STEP 6: Save outliers to CSV (optional)
# ------------------------------------
import os
os.makedirs("outputs", exist_ok=True)

for city, data in outliers.items():
    if not data.empty:
        data.to_csv(f"outputs/{city}_iqr_outliers.csv", index=False)

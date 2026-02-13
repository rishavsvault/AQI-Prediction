import pandas as pd
import numpy as np

file_path = "data/Air_quality_data.csv"
df = pd.read_csv(file_path)

# Convert Datetime if exists
if "Datetime" in df.columns:
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values(["City", "Datetime"])

# Select only numeric columns EXCEPT identifiers if any
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Treat only pollutant + AQI columns (not IDs)
columns_to_fix = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","AQI"]
columns_to_fix = [col for col in columns_to_fix if col in df.columns]

# Replace 0 with NaN ONLY in selected columns
for col in columns_to_fix:
    df[col] = df[col].replace(0, np.nan)

# Interpolate safely city-wise
for city in df["City"].unique():
    city_mask = df["City"] == city
    
    if "Datetime" in df.columns:
        df.loc[city_mask, columns_to_fix] = (
            df.loc[city_mask]
            .set_index("Datetime")[columns_to_fix]
            .interpolate(method="time")
            .ffill()
            .bfill()
            .values
        )
    else:
        df.loc[city_mask, columns_to_fix] = (
            df.loc[city_mask, columns_to_fix]
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .values
        )

# Save corrected file
output_path = "data/Air_quality_data_interpolated.csv"
df.to_csv(output_path, index=False)

print("Saved to:", output_path)

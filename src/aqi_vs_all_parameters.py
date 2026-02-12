import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
data = pd.read_csv("data/Air_quality_data.csv")

# Convert datetime
data["Datetime"] = pd.to_datetime(data["Datetime"])

# Filter 2015–2023
data = data[
    (data["Datetime"].dt.year >= 2015) &
    (data["Datetime"].dt.year <= 2023)
]

# Define only required cities
cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore"]

# Pollutants list
pollutants = [
    "PM2.5", "PM10", "NO", "NO2",
    "NOx", "NH3", "CO", "SO2", "O3"
]

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

for city in cities:

    city_df = data[data["City"] == city]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, pollutant in enumerate(pollutants):
        axes[i].scatter(city_df[pollutant], city_df["AQI"], alpha=0.4)
        axes[i].set_xlabel(pollutant)
        axes[i].set_ylabel("AQI")
        axes[i].set_title(f"AQI vs {pollutant}")
        axes[i].grid(True)

    plt.suptitle(f"{city} - AQI vs All Pollutants (2015-2023)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"outputs/{city}_AQI_vs_all_parameters.png")
    plt.close()

print("All 5 city plots generated successfully!")

import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# SETUP OUTPUT DIRECTORY
# -------------------------------------------------
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
monthly_aqi_df = pd.read_csv("data/Air_quality_data.csv")

# Standardize column naming based on your usage
# Note: Using 'Datetime' as the base column
monthly_aqi_df["Datetime"] = pd.to_datetime(monthly_aqi_df["Datetime"])
monthly_aqi_df["Date"] = monthly_aqi_df["Datetime"] # Alias for consistency

# -------------------------------------------------
# 1. MONTHLY AQI LINE GRAPH
# -------------------------------------------------
selected_cities = ["Delhi", "Kolkata", "Mumbai", "Bangalore"]
plt.figure(figsize=(10, 6))

for city in selected_cities:
    city_data = monthly_aqi_df[monthly_aqi_df["City"] == city].sort_values("Date")
    plt.plot(city_data["Date"], city_data["AQI"], label=city)

plt.xlabel("Date")
plt.ylabel("AQI")
plt.title("Monthly AQI Trend (Selected Cities)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/monthly_aqi_trend.png")
plt.show(block=False)

# -------------------------------------------------
# 2. YEARLY AVERAGE AQI
# -------------------------------------------------
monthly_aqi_df["Year"] = monthly_aqi_df["Date"].dt.year
yearly_avg_aqi = monthly_aqi_df.groupby(["City", "Year"])["AQI"].mean().reset_index()

plt.figure(figsize=(12, 7))
for city in yearly_avg_aqi["City"].unique():
    city_data = yearly_avg_aqi[yearly_avg_aqi["City"] == city]
    plt.plot(city_data["Year"], city_data["AQI"], marker="o", label=city)

plt.xlabel("Year")
plt.ylabel("Average AQI")
plt.title("Yearly Average AQI Trend (All Cities)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/yearly_avg_aqi.png")
plt.show()

# -------------------------------------------------
# 3. MOST & LEAST POLLUTED CITIES
# -------------------------------------------------
city_avg_aqi = monthly_aqi_df.groupby("City")["AQI"].mean().reset_index()
city_avg_aqi_sorted = city_avg_aqi.sort_values(by="AQI", ascending=False)

most_polluted = city_avg_aqi_sorted.head(5)
least_polluted = city_avg_aqi_sorted.tail(5)

# Top 5 Most
plt.figure(figsize=(8, 5))
plt.bar(most_polluted["City"], most_polluted["AQI"], color='salmon')
plt.title("Top 5 Most Polluted Cities")
plt.savefig(f"{output_dir}/top_5_most_polluted.png")
plt.show()

# Top 5 Least
plt.figure(figsize=(8, 5))
plt.bar(least_polluted["City"], least_polluted["AQI"], color='skyblue')
plt.title("Top 5 Least Polluted Cities")
plt.savefig(f"{output_dir}/top_5_least_polluted.png")
plt.show()

# -------------------------------------------------
# 4. PIE CHART
# -------------------------------------------------
plt.figure(figsize=(8, 8))
plt.pie(
    city_avg_aqi["AQI"],
    labels=city_avg_aqi["City"],
    autopct=lambda pct: f"{int(round(pct / 100.0 * sum(city_avg_aqi['AQI'])))}",
    startangle=140
)
plt.title("AQI Contribution by City (Average AQI Values)")
plt.tight_layout()
plt.savefig(f"{output_dir}/aqi_pie_chart.png")
plt.show()

# -------------------------------------------------
# 5. SCATTER PLOTS (Multi-Pollutant)
# -------------------------------------------------
selected_city = "Chennai"
city_data = monthly_aqi_df[monthly_aqi_df["City"] == selected_city]
pollutants = ["PM2.5", "NO", "NO2", "CO"]

for pollutant in pollutants:
    if pollutant in city_data.columns:
        plt.figure(figsize=(7, 5))
        plt.scatter(city_data[pollutant], city_data["AQI"], alpha=0.6, c='forestgreen')
        plt.xlabel(pollutant)
        plt.ylabel("AQI")
        plt.title(f"AQI vs {pollutant} for {selected_city}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scatter_AQI_vs_{pollutant}.png")
        plt.show()
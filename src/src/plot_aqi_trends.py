import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv("..\data\ML_Project_monthly aqi (2015-2024).csv")

# Convert Month column to datetime
df["Month"] = pd.to_datetime(df["Month"])

# Set Month as index
df.set_index("Month", inplace=True)

# Create plot
plt.figure(figsize=(12, 6))

for city in df.columns:
    plt.plot(df.index, df[city], label=city)

plt.xlabel("Year")
plt.ylabel("Average AQI")
plt.title("Monthly AQI Trends of 5 Cities")
plt.legend()
plt.grid(True)

# Save graph as image
plt.savefig("../outputs/aqi_trends.png", dpi=300)

# Show graph locally
plt.show()

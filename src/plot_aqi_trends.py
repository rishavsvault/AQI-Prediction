import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/ML_Project_monthly aqi (2015-2024).csv")

# Convert Month to datetime
df["Month"] = pd.to_datetime(df["Month"])

# Set Month as index
df.set_index("Month", inplace=True)

# Plot
plt.figure(figsize=(12, 6))

for city in df.columns:
    plt.plot(df.index, df[city], label=city)

plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.title("Monthly Average AQI of 5 Indian Cities")
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("aqi_trends.png", dpi=300)
plt.show()

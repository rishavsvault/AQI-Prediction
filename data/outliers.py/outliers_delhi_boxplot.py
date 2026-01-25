import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/daily_aqi_raw.csv")

df[["Delhi"]].boxplot(figsize=(10, 6))
plt.ylabel("AQI")
plt.title("Outlier Detection Using Boxplot")
plt.show()

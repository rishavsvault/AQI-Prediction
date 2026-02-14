import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load the dataset
df = pd.read_csv('data/Air_quality_data_interpolated.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# 2. Resample data
monthly_aqi = df.groupby('City')['AQI'].resample('ME').mean().reset_index()
pivot_df = monthly_aqi.pivot(index='Datetime', columns='City', values='AQI')

# 3. ENLARGE THE SCALE: figsize=(25, 12) makes it very wide and tall
plt.figure(figsize=(25, 12))

for city in pivot_df.columns:
    # marker='o' adds the circles
    # markersize sets the circle size
    # markerfacecolor='white' makes the dots "hollow" for better visibility
    plt.plot(pivot_df.index, pivot_df[city], 
             label=city, 
             linewidth=2, 
             marker='o', 
             markersize=6, 
             markerfacecolor='white', 
             markeredgewidth=1.5)

# 4. IMPROVE X-AXIS SCALE: Show every year explicitly
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Formatting
plt.title('Monthly Average AQI by City (2015 - 2024)', fontsize=22, pad=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Average AQI', fontsize=18)
plt.legend(title='City', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Rotate x-axis labels so they don't overlap at large scales
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# 5. Save with high resolution (DPI)
plt.savefig('plots/aqi_enlarged_with_dots.png', dpi=300, bbox_inches='tight')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load and prepare data
df = pd.read_csv('data/Air_quality_data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# 2. Resample to Monthly Average
monthly_aqi = df.groupby('City')['AQI'].resample('M').mean().reset_index()
pivot_df = monthly_aqi.pivot(index='Datetime', columns='City', values='AQI')

# 3. Setup the Large-Scale Subplots
cities = pivot_df.columns
num_cities = len(cities)
# 10 inches height per city to match your "original main plot" scale
fig, axes = plt.subplots(nrows=num_cities, ncols=1, figsize=(20, 10 * num_cities))

if num_cities == 1:
    axes = [axes]

# 4. Plotting
for i, city in enumerate(cities):
    axes[i].plot(pivot_df.index, pivot_df[city], 
                 label=f'{city} (2015-2024)', 
                 color=f'C{i}', 
                 linewidth=2.5, 
                 marker='o', 
                 markersize=6, 
                 markerfacecolor='white',
                 markeredgewidth=2)

    # TIMEFRAME in Title: Mentioning the years 2015-2024 specifically
    axes[i].set_title(f'{city}: Monthly AQI Trend (Timeframe: 2015 - 2024)', 
                      fontsize=24, fontweight='bold', pad=20)
    
    axes[i].set_ylabel('Average AQI', fontsize=18)
    axes[i].set_xlabel('Timeline (2015 - 2024)', fontsize=18)
    
    # Grid and X-Axis Formatting
    axes[i].grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Ensure every year is labeled on the X-axis
    axes[i].xaxis.set_major_locator(mdates.YearLocator()) 
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Set the x-axis limits strictly to your timeframe
    axes[i].set_xlim([pd.Timestamp('2015-01-01'), pd.Timestamp('2024-12-31')])
    
    axes[i].tick_params(axis='both', which='major', labelsize=14)
    axes[i].legend(fontsize=16, loc='upper right')

# 5. Final Layout Adjustments and Saving
plt.tight_layout(pad=6.0)

# Saving with the timeframe in the filename as well
output_filename = 'city_aqi_trends_2015_2024.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Success! The timeframe 2015-2024 is now explicitly mentioned in the titles, labels, and axis limits.")
plt.show()
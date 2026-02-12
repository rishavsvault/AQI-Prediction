import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# 1. Load the dataset
# Ensure the path matches your local file location
df = pd.read_csv('data/Air_quality_data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# 2. Define the metrics and the 5 specific cities
metrics = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore']

# 3. Process and Generate 5 PNGs
for city in cities:
    # Filter data for the specific city
    if city not in df['City'].unique():
        print(f"Skipping {city}: City name not found in CSV.")
        continue
        
    city_df = df[df['City'] == city]
    
    # Calculate Monthly Average for all columns
    # 'ME' stands for month-end frequency
    monthly_df = city_df[metrics].resample('ME').mean()
    
    # ENLARGE THE PLOT: Set to a massive scale for clarity
    plt.figure(figsize=(24, 12))
    
    # 4. Plot each pollutant/AQI as an overlapping line
    for metric in metrics:
        plt.plot(monthly_df.index, monthly_df[metric], 
                 label=metric, 
                 linewidth=2, 
                 marker='o',          # Circle markers
                 markersize=6, 
                 markerfacecolor='white', 
                 markeredgewidth=1.5,
                 alpha=0.8)           # Transparency to see overlaps

    # 5. Y-AXIS SCALE: Set a fixed difference of 25 units
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(25))
    
    # 6. Formatting for Timeframe and Clarity
    plt.title(f'{city}: Monthly Pollutant Trends (Timeframe: 2015 - 2024)', 
              fontsize=24, fontweight='bold', pad=25)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Value (Increments of 25)', fontsize=18)
    
    # X-Axis Timeframe formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlim([pd.Timestamp('2015-01-01'), pd.Timestamp('2024-12-31')])
    
    # Grid, Legend, and Ticks
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(title="Pollutants & AQI", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()

    # 7. Save the individual PNG for the city
    save_filename = f'{city}_AQI_Full_Trends_2015_2024.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close() # Important: Closes the current plot to free up memory for the next city

print("\nProcess Complete. 5 PNG files have been generated with a Y-axis interval of 25.")
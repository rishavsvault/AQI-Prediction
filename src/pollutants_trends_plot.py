import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# 1. Load the dataset
df = pd.read_csv('data/Air_quality_data_interpolated.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
target_cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore']

# 2. Process each pollutant
for pollutant in pollutants:
    # EXTRA LARGE CANVAS for maximum "de-clustering"
    plt.figure(figsize=(26, 12), facecolor='white')
    
    for i, city in enumerate(target_cities):
        city_data = df[df['City'] == city].copy()
        city_data.set_index('Datetime', inplace=True)
        
        # Calculate Monthly Average
        monthly_avg = city_data[pollutant].resample('M').mean()
        
        # Plotting with hollow markers and thinner lines
        plt.plot(monthly_avg.index, monthly_avg, 
                 label=city, 
                 linewidth=1.8, 
                 marker='o', 
                 markersize=5, 
                 markerfacecolor='white', # Hollow effect
                 markeredgewidth=1.5,
                 alpha=0.9) # Slight transparency to help at intersection points

    # 3. Y-Axis at strict 25-unit intervals
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(25))
    
    # 4. Clean Formatting
    plt.title(f'Comparative Analysis: {pollutant} Levels (2015 - 2024)', 
              fontsize=22, fontweight='bold', color='#333333', pad=30)
    plt.xlabel('Year', fontsize=16, labelpad=15)
    plt.ylabel(f'{pollutant} Concentration', fontsize=16, labelpad=15)
    
    # X-Axis Timeframe formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlim([pd.Timestamp('2015-01-01'), pd.Timestamp('2024-12-31')])
    
    # Remove top and right chart borders (spines) for a modern look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 5. Legend & Grid
    # Putting legend in a dedicated space to the right
    plt.legend(title="Cities", fontsize=13, title_fontsize=15, 
               bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Use a tight layout but provide room for the external legend
    plt.tight_layout()

    # 6. Save the PNG
    plt.savefig(f'{pollutant}_trends_2015_2024.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Clean overlapping graphs generated for all pollutants.")
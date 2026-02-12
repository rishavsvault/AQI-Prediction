import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_directory, 'Air_quality_data.csv')

df = pd.read_csv(file_path)

print("File loaded successfully!")

df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df[df['Datetime'].dt.year <= 2024]

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
city_profile = df.groupby('City')[features].mean()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(city_profile)

kmeans = KMeans(n_clusters=3, random_state=42)
city_profile['Cluster'] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                      c=city_profile['Cluster'], cmap='viridis', s=100)

for i, city in enumerate(city_profile.index):
    plt.annotate(city, (principal_components[i, 0], principal_components[i, 1]), 
                 xytext=(5, 5), textcoords='offset points')

plt.title('City Clusters based on Pollution Profile (Up to 2024)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Group')
plt.grid(True)
plt.show()

print("Final Clustering Groups:")
print(city_profile['Cluster'].sort_values())
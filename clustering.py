#Code was generated using ChatGPT (GPT-4; OpenAI).
#K-means clustering with dynamic time warping

import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import pandas as pd  

# Load your data
my_data = pd.read_excel('Q4 for DTW.xlsx') 

# Standardize the time series data
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # z-normalization
data_scaled = scaler.fit_transform(data_for_clustering)

# Define the range of clusters to try
cluster_range = range(1, 10)

# Calculate inertia for each k value
inertias = []
for k in cluster_range:
    kmeans_model = TimeSeriesKMeans(n_clusters=k, metric="dtw", verbose=False)
    kmeans_model.fit(data_scaled)
    inertias.append(kmeans_model.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
#plt.show()

# Save the figure with high resolution
output_figure_path = 'elbow.png'
plt.savefig(output_figure_path, dpi=600)
 
# Define the number of clusters
n_clusters = 4
random_seed = 10011

# Perform K-means clustering with DTW
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=False,
                         random_state=random_seed, n_init=10)
clusters = model.fit_predict(data_scaled)

# Add the cluster assignments to the original DataFrame
my_data['Cluster'] = clusters

# Save the DataFrame with cluster assignments to a CSV file
output_csv_path = '4cluster_assignments.csv'
my_data.to_csv(output_csv_path, index=False)

# Visualization of clusters
time_points = ['Baseline', 'T1', 'T2', 'T3']
plt.figure(figsize=(12, 8))

for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, yi + 1)
    for xx in data_scaled[clusters == yi]:
        # Ensure we only plot the actual data points
        plt.plot(xx[:4].ravel(), "k-", alpha=0.05, linewidth=0.1)  
    # Plot the cluster center for only the actual data points
    plt.plot(model.cluster_centers_[yi][:4].ravel(), "r-", label='Cluster Center')
    plt.xticks(ticks=np.arange(len(time_points)), labels=time_points)
    plt.title(f"Cluster {yi + 1}")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Save the figure with high resolution
output_figure_path = 'my4clusters.png'
plt.savefig(output_figure_path, dpi=600)

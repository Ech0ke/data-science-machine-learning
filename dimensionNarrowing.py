import pandas as pd
import plotly.express as px
import locale
import umap
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns

# Set the locale to use thousands separators
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
pd.options.display.float_format = '{:.2f}'.format

# Load the CSV data into object
data = pd.read_csv('global-data-on-sustainable-energy.csv')

columns_to_remove = [
    'Latitude',
    'Longitude',
    'Renewable-electricity-generating-capacity-per-capita',
    'Financial flows to developing countries (US $)',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
    f'Renewables (% equivalent primary energy)'
]
data.drop(columns=columns_to_remove, inplace=True)

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

filtered_data = data[data['Year'].isin([2014, 2019])].copy()
filtered_data['Density\\n(P/Km2)'] = pd.to_numeric(
    filtered_data['Density\\n(P/Km2)'].str.replace(r'[, ]', '', regex=True))

# Write the filtered data to a new CSV file
filtered_data.to_csv('filtered-data.csv', index=False)

# Task 2: descriptive analysis
with open('general_descriptive_analysis.txt', 'w') as f:
    f.write(filtered_data.describe().to_string())

grouped_filtered_data = filtered_data.groupby('Year')

# Calculate and write the summary statistics for each group to separate files
for year, group in grouped_filtered_data:
    year_str = str(year)
    with open(f'{year_str}_descriptive_analysis.txt', 'w') as f:
        f.write(f"Summary Statistics for Year {year}:\n")
        f.write(group.describe().to_string())

normalized_data_min_max = filtered_data.copy()

numeric_columns = ['Year', f'Access to electricity (% of population)', 'Access to clean fuels for cooking',
                   'Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)',
                   'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)', f'Low-carbon electricity (% electricity)',
                   'Primary energy consumption per capita (kWh/person)', 'Value_co2_emissions_kt_by_country',
                   'gdp_growth', 'gdp_per_capita', 'Density\\n(P/Km2)', 'Land Area(Km2)']

min_vals = filtered_data[numeric_columns].min()
max_vals = filtered_data[numeric_columns].max()
normalized_data_min_max[numeric_columns] = (
    filtered_data[numeric_columns] - min_vals) / (max_vals - min_vals)

normalized_data_min_max.to_csv('normalized.csv', index=False)

# Adjust marker size for better visibility


def adjust_colour_and_show(figure, title):
    figure.update_traces(marker=dict(size=5))
    figure.update_layout(title=title)
    figure.update_layout(plot_bgcolor="#d0d0e1")
    figure.show()


# Data preparation for dimension narrowing algorithms
filtered_data_grouped_by_country = filtered_data.groupby(
    "Entity", as_index=False).mean()
filtered_data_grouped_by_country.to_csv(
    'filtered_data_grouped_by_country.csv', index=False)

normalized_data_grouped_by_country = normalized_data_min_max.groupby(
    "Entity", as_index=False).mean()

# Task 2 select columns for clustering
columns_for_clustering_1 = [
    'Renewable energy share in the total final energy consumption (%)',
    'Electricity from fossil fuels (TWh)',
    'Electricity from nuclear (TWh)',
    'Electricity from renewables (TWh)']

columns_for_clustering_2 = [
    f'Low-carbon electricity (% electricity)', 'Electricity from renewables (TWh)', 'Primary energy consumption per capita (kWh/person)', f'Access to electricity (% of population)']

columns_for_clustering_3 = [f'Access to electricity (% of population)', 'Access to clean fuels for cooking',
                            'Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)',
                            'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)', f'Low-carbon electricity (% electricity)',
                            'Primary energy consumption per capita (kWh/person)', 'Value_co2_emissions_kt_by_country',
                            'gdp_growth', 'gdp_per_capita', 'Density\\n(P/Km2)', 'Land Area(Km2)']

# Data subset for clustering
clustering_data_1 = normalized_data_grouped_by_country[columns_for_clustering_1]
clustering_data_2 = normalized_data_grouped_by_country[columns_for_clustering_2]
clustering_data_3 = normalized_data_grouped_by_country[columns_for_clustering_3]

# Silhouette method
K = range(2, 20)


def optimal_clusters_silhouette(clustering_data, clustering_columns, K):
    sil_score = []
    for i in K:
        labels = cluster.KMeans(
            n_clusters=i, init="k-means++", n_init=10, random_state=200).fit(clustering_data).labels_
        score = metrics.silhouette_score(
            clustering_data, labels, metric="euclidean", sample_size=1000, random_state=200)
        sil_score.append(score)
        print("Silhouette score for k(clusters) = "+str(i)+" is "
              + str(metrics.silhouette_score(clustering_data, labels, metric="euclidean", sample_size=1000, random_state=200)))

    sil_centers = pd.DataFrame({'Clusters': K, 'Sil Score': sil_score})
    # Plot silhouette scores for different numbers of clusters
    plt.plot(sil_centers['Clusters'], sil_centers['Sil Score'], marker='o')
    columns_string = ', '.join(clustering_columns)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sil score')
    plt.title(
        f'Silhouette method for Optimal Cluster Number for columns {columns_string}')
    plt.show()


optimal_clusters_silhouette(clustering_data_1, columns_for_clustering_1, K)
optimal_clusters_silhouette(clustering_data_2, columns_for_clustering_2, K)
optimal_clusters_silhouette(clustering_data_3, columns_for_clustering_3, K)

# Elbow mehthod


def optimal_clusters_elbow(clustering_data, clustering_columns, K):
    wss = []
    for k in K:
        # Set n_init explicitly
        kmeans = cluster.KMeans(n_clusters=k, init="k-means++", n_init=10)
        kmeans = kmeans.fit(clustering_data)
        wss_iter = kmeans.inertia_
        wss.append(wss_iter)
        print("Wss score for k(clusters) = " + str(k) + " is "
              + str(wss_iter))

    mycenters = pd.DataFrame({'Clusters': K, 'WSS': wss})
    # Plot the wss scores
    plt.plot(mycenters['Clusters'], mycenters['WSS'], marker='o')
    columns_string = ', '.join(clustering_columns)
    plt.xlabel('Number of clusters')
    plt.ylabel('WSS score')
    plt.title(
        f'Elbow Method for Optimal Cluster Number for columns {columns_string}')
    plt.show()


optimal_clusters_elbow(clustering_data_1, columns_for_clustering_1, K)
optimal_clusters_elbow(clustering_data_2, columns_for_clustering_2, K)
optimal_clusters_elbow(clustering_data_3, columns_for_clustering_3, K)


def plot_correlation_heatmap(data, title):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))  # Set the figure size to your preference

    plt.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Correlation')

    # Set ticks and labels with smaller fonts
    ticks = range(len(correlation_matrix.columns))
    plt.xticks(ticks, correlation_matrix.columns,
               rotation=45, fontsize=8)  # Adjust fontsize
    plt.yticks(ticks, correlation_matrix.columns,
               fontsize=8)  # Adjust fontsize

    # Add correlation values to the heatmap
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            if i == j:
                continue
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                     ha='center', va='center', color='#303030', fontsize=6)  # Adjust fontsize

    # Display the plot
    plt.title(title)
    plt.tight_layout()  # Ensure tight layout
    plt.show()


plot_correlation_heatmap(clustering_data_1, "Correlation Matrix - Data 1")
plot_correlation_heatmap(clustering_data_2, "Correlation Matrix - Data 2")
plot_correlation_heatmap(clustering_data_3, "Correlation Matrix - Data 3")

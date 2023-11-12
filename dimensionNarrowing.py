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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
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
columns_for_clustering_1 = [f'Access to electricity (% of population)',
                            'Renewable energy share in the total final energy consumption (%)',
                            'Electricity from fossil fuels (TWh)',
                            'Electricity from nuclear (TWh)',
                            'Electricity from renewables (TWh)']

columns_for_clustering_2 = [
    f'Renewable energy share in the total final energy consumption (%)', 'Electricity from renewables (TWh)', 'Primary energy consumption per capita (kWh/person)', 'Value_co2_emissions_kt_by_country']

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


# Dendogram
def plot_dendrogram(model, clustering_columns, lineHeight, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)
    plt.axhline(y=lineHeight, color='black', linestyle='--')
    columns_string = '\n'.join(clustering_columns)
    plt.title(f"Hierarchical Clustering Dendrogram for columns {columns_string}", pad=1)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(clustering_data_1)
# plot_dendrogram(model, columns_for_clustering_1, 0.75, truncate_mode="level")
# model = model.fit(clustering_data_2)
# plot_dendrogram(model, columns_for_clustering_2, 0.75, truncate_mode="level")
# model = model.fit(clustering_data_3)
# plot_dendrogram(model, columns_for_clustering_3, 1, truncate_mode="level")


def kmeans_clustering(data, n_clusters, title):
    umap_data = data
    kmeans = cluster.KMeans(n_clusters= n_clusters, n_init=10, random_state=10).fit(umap_data)
    reduced_data_umap = umap.UMAP(
        n_components=2, random_state=42).fit_transform(umap_data)
    kmeans_df = pd.DataFrame(reduced_data_umap, columns=["x", "y"])
    kmeans_df["Cluster"] = kmeans.labels_
    kmeans_df["Valstybė"] = normalized_data_grouped_by_country["Entity"]
    fig = px.scatter(kmeans_df, x="x", y="y", color="Cluster", title=title, hover_name="Valstybė")
    fig.show()

kmeans_clustering(clustering_data_1, 5, "UMAP Visualization - Data 1 (K-Means)")
kmeans_clustering(clustering_data_2, 6, "UMAP Visualization - Data 2 (K-Means)")
kmeans_clustering(clustering_data_3, 6, "UMAP Visualization - Data 3 (K-Means)")

# # UMAP for filtered data
# umap_data = filtered_data_grouped_by_country.drop(
#     columns=["Entity", "Access to electricity (% of population)"])

# reduced_data_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(
#     umap_data)

# umap_df = pd.DataFrame(reduced_data_umap, columns=["x", "y"])

# # Add back misisng columns for data visualization
# umap_df["Valstybė"] = filtered_data_grouped_by_country["Entity"]
# umap_df["Prieiga prie elektros (% nuo populiacijos)"] = filtered_data_grouped_by_country["Access to electricity (% of population)"]

# # Use Plotly Express for interactive plotting for filtered data
# fig_electricity_percentage = px.scatter(umap_df, x="x", y="y", range_x=[-8, 15], range_y=[-6, 8],
#                                         color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

# adjust_colour_and_show(fig_electricity_percentage, "UMAP projekcija")

# # UMAP for normalized data
# umap_data_normalized = normalized_data_grouped_by_country.drop(
#     columns=["Entity", "Access to electricity (% of population)"])

# reduced_data_umap_normalized = umap.UMAP(n_components=2, random_state=42).fit_transform(
#     umap_data_normalized)

# umap_df_normalized = pd.DataFrame(
#     reduced_data_umap_normalized, columns=["x", "y"])

# # Add back misisng columns for data visualization
# umap_df_normalized["Valstybė"] = normalized_data_grouped_by_country["Entity"]
# umap_df_normalized["Prieiga prie elektros (% nuo populiacijos)"] = normalized_data_grouped_by_country[
#     "Access to electricity (% of population)"]

# # Use Plotly Express for interactive plotting for normalized data
# fig_electricity_percentage_normalized = px.scatter(umap_df_normalized, x="x", y="y", range_x=[-1, 14], range_y=[-1, 14],
#                                                    color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

# adjust_colour_and_show(fig_electricity_percentage_normalized,
#                        "UMAP projekcija (normalizuoti duomenys)")

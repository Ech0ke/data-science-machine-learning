import pandas as pd
import plotly.express as px
import locale
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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


def optimal_clusters_silhouette(clustering_data, clustering_columns):
    silhouette_scores = []  # Store silhouette scores
    cluster_numbers = []  # Store cluster numbers

    for num_clusters in range(2, 20):
        clusterer = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = clusterer.fit_predict(clustering_data)
        silhouette_avg = silhouette_score(clustering_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        cluster_numbers.append(num_clusters)

    # Find the optimal number of clusters based on the highest silhouette score
    optimal_cluster_num = cluster_numbers[silhouette_scores.index(
        max(silhouette_scores))]
    columns_string = ', '.join(clustering_columns)
    print(
        f"For data: {columns_string}\nOptimal number of clusters suggested by Silhouette Method:", optimal_cluster_num)


optimal_clusters_silhouette(clustering_data_1, columns_for_clustering_1)
optimal_clusters_silhouette(clustering_data_2, columns_for_clustering_2)
optimal_clusters_silhouette(clustering_data_3, columns_for_clustering_3)

# Elbow mehthod


def optimal_clusters_elbow(clustering_data, clustering_columns):
    # Convert data to pairwise distances
    distance_matrix = pdist(clustering_data)

    # Calculate linkage matrix
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Calculate inertia (cophenetic distance) for different numbers of clusters
    inertia = []
    for num_clusters in range(2, 20):
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        # Calculate the inertia within each cluster
        inertia_sum = 0
        for cluster_num in range(1, num_clusters + 1):
            cluster_indices = (clusters == cluster_num)
            cluster_data = clustering_data[cluster_indices]
            inertia_sum += pdist(cluster_data).var()
        inertia.append(inertia_sum)

    # Plot the inertia
    plt.plot(range(2, 20), inertia, marker='o')
    columns_string = ', '.join(clustering_columns)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title(
        f'Elbow Method for Optimal Cluster Number  for columns {columns_string}')
    plt.show()


optimal_clusters_elbow(clustering_data_1, columns_for_clustering_1)
optimal_clusters_elbow(clustering_data_2, columns_for_clustering_2)
optimal_clusters_elbow(clustering_data_3, columns_for_clustering_3)

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

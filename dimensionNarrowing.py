import pandas as pd
import plotly.express as px
import locale
import umap
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score as ss
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.cluster import DBSCAN
import itertools
import numpy as np

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

# Constant for trying different cluster numbers in finding optimal cluster count
K = range(2, 20)


# Silhouette method
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

# Task 4 data clustering


# DB scan clustering
def dbscan_clustering(data, eps, min_samples):
    X = data.to_numpy()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    clusters = dbscan.labels_
    return clusters


def plot_umap_with_clusters(data, columns, title, eps, min_samples):
    umap_data = data
    clusters = dbscan_clustering(umap_data, eps, min_samples)

    reduced_data_umap = umap.UMAP(
        n_components=2, random_state=42).fit_transform(umap_data)
    umap_df = pd.DataFrame(reduced_data_umap, columns=["x", "y"])
    umap_df["Cluster"] = clusters
    print(f'Silhouette score of dbscan: {ss(umap_data, umap_df["Cluster"])}')

    fig = px.scatter(umap_df, x="x", y="y", color="Cluster", title=title)
    fig.show()


epsilons = np.linspace(0.01, 1, num=15)
min_samples = np.arange(2, 20, step=3)
combinations = list(itertools.product(epsilons, min_samples))
N = len(combinations)


def get_scores_and_labels(combinations, X):
    scores = []
    all_labels_list = []

    for i, (eps, num_samples) in enumerate(combinations):
        dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
        labels = dbscan_cluster_model.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)
        if -1 in labels_set:
            num_clusters -= 1

        if (num_clusters < 2) or (num_clusters > 50):
            scores.append(-10)
            all_labels_list.append('bad')
            c = (eps, num_samples)
            continue

        scores.append(ss(X, labels))
        all_labels_list.append(labels)

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]

    return {'best_epsilon': best_parameters[0],
            'best_min_samples': best_parameters[1],
            'best_labels': best_labels,
            'best_score': best_score}


best_dict_1 = get_scores_and_labels(combinations, clustering_data_1.to_numpy())
print(f'Best DBSCAN parameters for columns_for_clustering_1: {best_dict_1}')
best_dict_2 = get_scores_and_labels(combinations, clustering_data_2.to_numpy())
print(f'Best DBSCAN parameters for columns_for_clustering_2: {best_dict_2}')
best_dict_3 = get_scores_and_labels(combinations, clustering_data_3.to_numpy())
print(f'Best DBSCAN parameters for columns_for_clustering_3: {best_dict_3}')

# # Plot clustering results
plot_umap_with_clusters(clustering_data_1, columns_for_clustering_1,
                        "UMAP Visualization - Data 1", eps=0.22, min_samples=14)
plot_umap_with_clusters(clustering_data_2, columns_for_clustering_2,
                        "UMAP Visualization - Data 2", eps=0.08, min_samples=11)
plot_umap_with_clusters(clustering_data_3, columns_for_clustering_3,
                        "UMAP Visualization - Data 3", eps=0.65, min_samples=2)

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

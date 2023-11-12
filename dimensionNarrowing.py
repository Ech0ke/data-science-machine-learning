import pandas as pd
import plotly.express as px
import locale
import umap
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score as ss
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
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
plot_dendrogram(model, columns_for_clustering_1, 0.75, truncate_mode="level")
model = model.fit(clustering_data_2)
plot_dendrogram(model, columns_for_clustering_2, 0.75, truncate_mode="level")
model = model.fit(clustering_data_3)
plot_dendrogram(model, columns_for_clustering_3, 1, truncate_mode="level")


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

kmeans_clustering(clustering_data_1, 7, "UMAP Visualization - Data 1 (K-Means)")
kmeans_clustering(clustering_data_2, 10, "UMAP Visualization - Data 2 (K-Means)")
kmeans_clustering(clustering_data_3, 6, "UMAP Visualization - Data 3 (K-Means)")

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
    umap_df["Valstybė"] = normalized_data_grouped_by_country["Entity"]
    print(f'Silhouette score of dbscan: {ss(umap_data, umap_df["Cluster"])}')

    fig = px.scatter(umap_df, x="x", y="y", color="Cluster",
                     title=title, hover_name="Valstybė")
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

# Plot clustering results
plot_umap_with_clusters(clustering_data_1, columns_for_clustering_1,
                        "UMAP Visualization - Data 1", eps=0.081, min_samples=14)
plot_umap_with_clusters(clustering_data_2, columns_for_clustering_2,
                        "UMAP Visualization - Data 2", eps=0.22, min_samples=8)
plot_umap_with_clusters(clustering_data_3, columns_for_clustering_3,
                        "UMAP Visualization - Data 3", eps=0.65, min_samples=2)

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


# plot_correlation_heatmap(clustering_data_1, "Correlation Matrix - Data 1")
# plot_correlation_heatmap(clustering_data_2, "Correlation Matrix - Data 2")
# plot_correlation_heatmap(clustering_data_3, "Correlation Matrix - Data 3")

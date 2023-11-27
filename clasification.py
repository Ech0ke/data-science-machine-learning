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

# Clear rows containing empty values
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

filtered_data = data[data['Year'] == 2019].copy()
filtered_data['Density\\n(P/Km2)'] = pd.to_numeric(
    filtered_data['Density\\n(P/Km2)'].str.replace(r'[, ]', '', regex=True))

# Drop year column since it is constant now
filtered_data.drop(['Year'], axis=1, inplace=True)

# Write the filtered data to a new CSV file
filtered_data.to_csv('filtered-data-2019.csv', index=False)

# Task 1

# Save country names into 2 separate datasets
up_to_90_countries = filtered_data[filtered_data['Access to electricity (% of population)'] <= 90]['Entity'].copy(
)
above_90_countries = filtered_data[filtered_data['Access to electricity (% of population)'] > 90]['Entity'].copy(
)

normalized_data = filtered_data.copy()

numeric_columns = [f'Access to electricity (% of population)', 'Access to clean fuels for cooking',
                   'Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)',
                   'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)', f'Low-carbon electricity (% electricity)',
                   'Primary energy consumption per capita (kWh/person)', 'Value_co2_emissions_kt_by_country',
                   'gdp_growth', 'gdp_per_capita', 'Density\\n(P/Km2)', 'Land Area(Km2)']

min_vals = filtered_data[numeric_columns].min()
max_vals = filtered_data[numeric_columns].max()

# Normalize data
normalized_data[numeric_columns] = (
    filtered_data[numeric_columns] - min_vals) / (max_vals - min_vals)

normalized_data.to_csv('normalized.csv', index=False)

up_to_90_data_normalized = pd.merge(
    up_to_90_countries,
    normalized_data,
    on='Entity',
    how='inner'  # Left join to retain entities that match up_to_90_countries
).copy()

above_90_data_normalized = pd.merge(
    above_90_countries,
    normalized_data,
    on='Entity',
    how='inner'  # Left join to retain entities that match up_to_90_countries
).copy()

up_to_90_data_normalized.to_csv('up-to-90-data-normalized.csv', index=False)
above_90_data_normalized.to_csv('above-90-data-normalized.csv', index=False)


def adjust_colour_and_show(figure, title):
    figure.update_traces(marker=dict(size=5))
    figure.update_layout(title=title)
    figure.update_layout(plot_bgcolor="#d0d0e1")
    figure.show()


# Umap data set
# Need this, because without it umap visualization loses almost all countries, idk why. This doesn't change actual data, just reorders it
normalized_data_grouped_by_country = normalized_data.groupby(
    "Entity", as_index=False).mean()
umap_data_normalized = normalized_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_umap_normalized = umap.UMAP(n_components=2, random_state=42).fit_transform(
    umap_data_normalized)

umap_df_normalized = pd.DataFrame(
    reduced_data_umap_normalized, columns=["x", "y"])

# Add back misisng columns for data visualization
umap_df_normalized["Entity"] = normalized_data_grouped_by_country["Entity"]
umap_df_normalized["Access to electricity (% of population)"] = normalized_data_grouped_by_country[
    "Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for normalized data
fig_electricity_percentage_normalized = px.scatter(umap_df_normalized, x="x", y="y", range_x=[-1, 14], range_y=[-1, 14],
                                                   color="Access to electricity (% of population)", hover_name="Entity")

adjust_colour_and_show(fig_electricity_percentage_normalized,
                       "UMAP projekcija (normalizuoti duomenys)")


# Task 2

# Descriptive analysis
up_to_90_data = filtered_data[filtered_data['Access to electricity (% of population)'] <= 90].copy(
)
above_90_data = filtered_data[filtered_data['Access to electricity (% of population)'] > 90].copy(
)

up_to_90_data.to_csv('up-to-90-data-2019.csv', index=False)
above_90_data.to_csv('above-90-data-2019.csv', index=False)

with open('general_descriptive_analysis.txt', 'w') as f:
    f.write(filtered_data.describe().to_string())

with open('up_to_90_analysis.txt', 'w') as f:
    f.write(up_to_90_data.describe().to_string())

with open('above_90_analysis.txt', 'w') as f:
    f.write(above_90_data.describe().to_string())

# Task 3

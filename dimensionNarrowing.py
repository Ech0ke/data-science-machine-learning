import pandas as pd
import plotly.express as px
import locale
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# Task 1: select data to analyse
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

# Task 3: Min-Max normalization for numeric columns
min_vals = filtered_data[numeric_columns].min()
max_vals = filtered_data[numeric_columns].max()
normalized_data_min_max[numeric_columns] = (
    filtered_data[numeric_columns] - min_vals) / (max_vals - min_vals)

normalized_data_min_max.to_csv('normalized.csv', index=False)

# Task 4: dimension narrowing

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

# UMAP for filtered data
umap_data = filtered_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(
    umap_data)

umap_df = pd.DataFrame(reduced_data_umap, columns=["x", "y"])

# Add back misisng columns for data visualization
umap_df["Valstybė"] = filtered_data_grouped_by_country["Entity"]
umap_df["Prieiga prie elektros (% nuo populiacijos)"] = filtered_data_grouped_by_country["Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for filtered data
fig_electricity_percentage = px.scatter(umap_df, x="x", y="y", range_x=[-8, 15], range_y=[-6, 8],
                                        color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

adjust_colour_and_show(fig_electricity_percentage, "UMAP projekcija")

normalized_data_grouped_by_country = normalized_data_min_max.groupby(
    "Entity", as_index=False).mean()

# UMAP for normalized data
umap_data_normalized = normalized_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_umap_normalized = umap.UMAP(n_components=2, random_state=42).fit_transform(
    umap_data_normalized)

umap_df_normalized = pd.DataFrame(
    reduced_data_umap_normalized, columns=["x", "y"])

# Add back misisng columns for data visualization
umap_df_normalized["Valstybė"] = normalized_data_grouped_by_country["Entity"]
umap_df_normalized["Prieiga prie elektros (% nuo populiacijos)"] = normalized_data_grouped_by_country[
    "Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for normalized data
fig_electricity_percentage_normalized = px.scatter(umap_df_normalized, x="x", y="y", range_x=[-1, 14], range_y=[-1, 14],
                                                   color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

adjust_colour_and_show(fig_electricity_percentage_normalized, "UMAP projekcija (normalizuoti duomenys)")

# PCA for filtered data
pca_data = filtered_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_pca = PCA(n_components=2).fit_transform(pca_data)

pca_df = pd.DataFrame(reduced_data_pca, columns=["x", "y"])

# Add back misisng columns for data visualization
pca_df["Valstybė"] = filtered_data_grouped_by_country["Entity"]
pca_df["Prieiga prie elektros (% nuo populiacijos)"] = filtered_data_grouped_by_country["Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for filtered data
fig_electricity_percentage = px.scatter(pca_df, x="x", y="y", range_x=[-800000, 1400000], range_y=[-600000, 800000],
                                        color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

adjust_colour_and_show(fig_electricity_percentage, "PCA projekcija")

# PCA for normalized data
pca_data_normalized = normalized_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_pca_normalized = PCA(
    n_components=2).fit_transform(pca_data_normalized)

pca_df_normalized = pd.DataFrame(
    reduced_data_pca_normalized, columns=["x", "y"])

# Add back misisng columns for data visualization
pca_df_normalized["Valstybė"] = normalized_data_grouped_by_country["Entity"]
pca_df_normalized["Prieiga prie elektros (% nuo populiacijos)"] = normalized_data_grouped_by_country[
    "Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for normalized data
fig_electricity_percentage_normalized = px.scatter(pca_df_normalized, x="x", y="y", range_x=[-1, 14], range_y=[-1, 14],
                                                   color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

adjust_colour_and_show(fig_electricity_percentage_normalized, "PCA projekcija (normalizuoti duomenys)")

# MDS for filtered data
mds_data = filtered_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

mds = MDS(n_components=2, dissimilarity='euclidean', random_state = 99)
reduced_data_mds = mds.fit_transform(mds_data)

mds_df = pd.DataFrame(reduced_data_mds, columns=["x", "y"])

# Add back missing columns for data visualization
mds_df["Valstybė"] = filtered_data_grouped_by_country["Entity"]
mds_df["Prieiga prie elektros (% nuo populiacijos)"] = filtered_data_grouped_by_country["Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for MDS data
fig_electricity_percentage_mds = px.scatter(mds_df, x="x", y="y", 
    color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

adjust_colour_and_show(fig_electricity_percentage_mds, "MDS projekcija")


# MDS for normalized data
mds_data_normalized = normalized_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

mds_normalized = MDS(n_components=2, dissimilarity='euclidean', random_state = 15)
reduced_data_mds_normalized = mds_normalized.fit_transform(mds_data_normalized)

mds_df_normalized = pd.DataFrame(reduced_data_mds_normalized, columns=["x", "y"])

# Add back missing columns for data visualization
mds_df_normalized["Valstybė"] = normalized_data_grouped_by_country["Entity"]
mds_df_normalized["Prieiga prie elektros (% nuo populiacijos)"] = normalized_data_grouped_by_country[
    "Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for MDS data
fig_electricity_percentage_mds_normalized = px.scatter(mds_df_normalized, x="x", y="y", range_x=[-1, 14], range_y=[-1, 14],
                                                       color="Prieiga prie elektros (% nuo populiacijos)", hover_name="Valstybė")

adjust_colour_and_show(fig_electricity_percentage_mds_normalized, "MDS projekcija (normalizuoti duomenys)")


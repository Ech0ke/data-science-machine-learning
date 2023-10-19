import pandas as pd
import plotly.express as px
import numpy as np
import locale
import matplotlib.pyplot as plt
import umap

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

# UMAP method

# filtered_data_grouped_by_country = filtered_data.groupby("Entity").mean()
filtered_data_grouped_by_country = filtered_data.groupby(
    "Entity", as_index=False).mean()
filtered_data_grouped_by_country.to_csv(
    'filtered_data_grouped_by_country.csv', index=False)


umap_data = filtered_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(
    umap_data)

umap_df = pd.DataFrame(reduced_data_umap, columns=["x", "y"])
umap_df["Entity"] = filtered_data_grouped_by_country["Entity"]
umap_df["Access to electricity (% of population)"] = filtered_data_grouped_by_country["Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting
fig_electricity_percentage = px.scatter(umap_df, x="x", y="y",
                                        color="Access to electricity (% of population)", hover_name="Entity")
# Adjust marker size for better visibility
fig_electricity_percentage.update_traces(marker=dict(size=5))
fig_electricity_percentage.update_layout(title="UMAP Projection")
fig_electricity_percentage.show()

normalized_data_grouped_by_country = normalized_data_min_max.groupby(
    "Entity", as_index=False).mean()

# UMAP for normalized data
umap_data_normalized = normalized_data_grouped_by_country.drop(
    columns=["Entity", "Access to electricity (% of population)"])

reduced_data_umap_normalized = umap.UMAP(n_components=2, random_state=42).fit_transform(
    umap_data_normalized)

umap_df_normalized = pd.DataFrame(
    reduced_data_umap_normalized, columns=["x", "y"])
umap_df_normalized["Entity"] = normalized_data_grouped_by_country["Entity"]
umap_df_normalized["Access to electricity (% of population)"] = normalized_data_grouped_by_country[
    "Access to electricity (% of population)"]

# Use Plotly Express for interactive plotting for normalized data
fig_electricity_percentage_normalized = px.scatter(umap_df_normalized, x="x", y="y",
                                                   color="Access to electricity (% of population)", hover_name="Entity")
# Adjust marker size for better visibility
fig_electricity_percentage_normalized.update_traces(marker=dict(size=5))
fig_electricity_percentage_normalized.update_layout(
    title="UMAP Projection (Normalized Data)")
fig_electricity_percentage_normalized.show()

import pandas as pd
import plotly.express as px
import numpy as np
import locale
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

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
filtered_data['Density\\n(P/Km2)'] = pd.to_numeric(filtered_data['Density\\n(P/Km2)'].str.replace(r'[, ]', '', regex=True))

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

# Min-Max normalization for numeric columns
min_vals = filtered_data[numeric_columns].min()
max_vals = filtered_data[numeric_columns].max()
normalized_data_min_max[numeric_columns] = (
    filtered_data[numeric_columns] - min_vals) / (max_vals - min_vals)

normalized_data_min_max.to_csv('normalized.csv', index=False)

# Principal Component Analysis (PCA) method

# Not normalized data
x = filtered_data.loc[:, numeric_columns].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, random_state=200, svd_solver="randomized")
principal_components = pca.fit_transform(x)

principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
final_df = pd.concat([principal_df, filtered_data['Entity'].reset_index(drop=True)], axis=1)

fig = px.scatter(final_df, x='principal component 1', y='principal component 2',
                color='Entity', hover_data=['Entity'], labels={'Entity': 'Valstybė', 'principal component 1': 'x', 'principal component 2': 'y'})
fig.update_layout(title='2 dimensijų PCA algoritmas', xaxis_title='x', yaxis_title='y')
fig.show()

# Normalized data
x = normalized_data_min_max.loc[:, numeric_columns].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2, random_state=200, svd_solver="randomized")
principal_components = pca.fit_transform(x)

principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
final_df = pd.concat([principal_df, filtered_data['Entity'].reset_index(drop=True)], axis=1)

fig = px.scatter(final_df, x='principal component 1', y='principal component 2',
                color='Entity', hover_data=['Entity'], labels={'Entity': 'Valstybė', 'principal component 1': 'x', 'principal component 2': 'y'})
fig.update_layout(title='2 dimensijų PCA algoritmas', xaxis_title='x', yaxis_title='y')
fig.show()
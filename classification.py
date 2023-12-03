import pandas as pd
import plotly.express as px
import locale
import umap
import numpy as np
from sklearn.metrics import silhouette_score as ss
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier  # Import K-NN classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set the locale to use thousands separators
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
pd.options.display.float_format = '{:.2f}'.format


# Constants
TARGET_COLUMN = 'Access to electricity (% of population)'
THRESHOLD = 0.9
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

# Task 1 create datasets for clasification

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


# Task 2 descriptive analysis

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

# Balance between classes
data_copy = normalized_data.copy()

# Assuming 'Label' column contains boolean values
data_copy['Label'] = data_copy[TARGET_COLUMN] >= THRESHOLD
# Map boolean values to strings
data_copy['Label'] = data_copy['Label'].replace(
    {False: 'Iki 90%', True: '90%-100%'})

# Create histogram with modified labels
spread = px.histogram(data_copy, x='Label', title='Klasės balansas')

# Update axis labels
spread.update_xaxes(title_text='Klasė')
spread.update_yaxes(title_text='Skaičius')

# Display counts above the bars
spread.update_traces(texttemplate='%{y}', textposition='outside')

spread.show()

# Task 3 split dataset into learn, validate, train
classification_data = normalized_data.copy()
test_data = above_90_data_normalized.sample(n=10)
test_data = pd.concat(
    [test_data, up_to_90_data_normalized.sample(n=10)], ignore_index=True)

entities_to_drop = test_data['Entity'].tolist()
classification_data = classification_data[~classification_data['Entity'].isin(
    entities_to_drop)]

train_data = classification_data.copy()

# Umap data split
umap_classification_data = umap_df_normalized.copy()
umap_test_data = umap_classification_data[umap_classification_data['Entity'].isin(
    test_data['Entity'])]

umap_classification_data = umap_classification_data[~umap_classification_data['Entity'].isin(
    entities_to_drop)]

# Define feature columns and target variable
feature_columns = [col for col in train_data.columns if col not in [
    'Access to electricity (% of population)', 'Entity', 'Label']]
feature_columns_umap = [col for col in umap_classification_data.columns if col not in [
    'Access to electricity (% of population)', 'Entity', 'Label']]

# Create binary labels based on the THRESHOLD
train_data['Label'] = (train_data[TARGET_COLUMN] >= THRESHOLD)
# validate_data['Label'] = (validate_data[TARGET_COLUMN] >= THRESHOLD)
test_data['Label'] = (test_data[TARGET_COLUMN] >= THRESHOLD)


'''
# Prepare data
X_train = train_data[feature_columns]
y_train = train_data['Label']

X_test = test_data[feature_columns]
y_test = test_data['Label']

# Define the hyperparameters you want to tune
param_grid = {
    # Example priors to try
    'priors': [None, [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.66, 0.34]],
    # Different values for var_smoothing
    'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
}

# Task 4: classify using Gaussian Naive Bayes
# Initialize Naive Bayes classifier
classifier = GaussianNB()

# Perform grid search with cross-validation
grid_search = GridSearchCV(classifier, param_grid,
                           cv=5)  # 5-fold cross-validation
grid_search.fit(X_train, y_train)

# Get the best parameters found by grid search
best_params = grid_search.best_params_
best_classifier = grid_search.best_estimator_

# Use the best classifier for final evaluation on test data
y_test_pred = best_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_rep = classification_report(y_test, y_test_pred)

print(f"Accuracy on test data: {test_accuracy}")
print("Classification Report on test data:\n", test_classification_rep)
print("Best parameters for the original dataset:")
print(best_params)

umap_classification_data['Label'] = (
    umap_classification_data[TARGET_COLUMN] >= THRESHOLD)

umap_test_data['Label'] = (umap_test_data[TARGET_COLUMN] >= THRESHOLD)

y_train_umap = umap_classification_data['Label']
X_train_umap = umap_classification_data[feature_columns_umap]

y_test_umap = umap_test_data['Label']
X_test_umap = umap_test_data[feature_columns_umap]

# Initialize Naive Bayes classifier
classifier = GaussianNB()

# Perform grid search with cross-validation
grid_search = GridSearchCV(classifier, param_grid,
                           cv=5)  # 5-fold cross-validation
grid_search.fit(X_train_umap, y_train_umap)

# Get the best parameters found by grid search
best_params_umap = grid_search.best_params_
best_classifier_umap = grid_search.best_estimator_

# Use the best classifier for final evaluation on test data
y_test_pred_umap = best_classifier_umap.predict(X_test_umap)
test_accuracy_umap = accuracy_score(y_test_umap, y_test_pred_umap)
test_classification_rep_umap = classification_report(
    y_test_umap, y_test_pred_umap)

print(f"Accuracy on test data umap: {test_accuracy_umap}")
print("Classification Report on test data for umap:\n",
      test_classification_rep_umap)
print("Best parameters for the UMAP dataset:")
print(best_params_umap)

# Fit the LabelEncoder for labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test_pred)
y_train_encoded_umap = label_encoder.fit_transform(y_train_umap)
y_test_encoded_umap = label_encoder.transform(y_test_pred_umap)

'''

def plot_decision_boundary(X, y, classifier, title, entity_df):
    # Perform UMAP for dimensionality reduction to 2 components
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X)

    # Fit the classifier on UMAP-transformed data
    classifier.fit(X_umap, y)

    # Create a meshgrid for decision boundary
    x_min, x_max = X_umap[:, 0].min() - 1, X_umap[:, 0].max() + 1
    y_min, y_max = X_umap[:, 1].min() - 1, X_umap[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict for each point in meshgrid to obtain the decision boundary
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Initialize an empty figure using Plotly Express
    fig = go.Figure()

    # Add contour plot for decision boundary (Trace 0)
    fig.add_contour(x=xx[0], y=yy[:, 0], z=Z, opacity=0.1,
                    colorscale='Fall', showlegend=False, showscale=False)

    entity_df = entity_df.groupby("Entity", as_index=False).mean()

    # Add scatter plot for UMAP-transformed data (Trace 1)
    scatter_trace = go.Scatter(x=X_umap[:, 0], y=X_umap[:, 1], mode='markers',
                               marker=dict(color=y, colorscale='Fall'),
                               legendgroup='group', showlegend=False)
    fig.add_trace(scatter_trace)

    # Add separate traces for each class for the legend
    classes = set(y)
    for label in classes:
        data_by_label = X_umap[y == label]
        legend_name = '90%-100%' if label == 1 else 'iki 90%'
        entity_texts = entity_df.loc[y == label, 'Entity']
        hover_text = [f'<b>Valstybė:</b> {entity}' for entity in entity_texts]
        fig.add_trace(go.Scatter(x=data_by_label[:, 0], y=data_by_label[:, 1], mode='markers',
                                 marker=dict(color=label, colorscale='Fall'),
                                 legendgroup=f'{label}', showlegend=True, name=legend_name,
                                 text=hover_text))

    # Update layout with labels and title
    fig.update_layout(title=title, xaxis_title='UMAP Component 1',
                      yaxis_title='UMAP Component 2')

    fig.show()


def plot_decision_boundary_umap(X, y, classifier, title, entity_df):

    # Fit the classifier on UMAP-transformed data
    classifier.fit(X, y)

    print(type(X))

    # Create a meshgrid for decision boundary
    x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
    y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict for each point in meshgrid to obtain the decision boundary
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Initialize an empty figure using Plotly Express
    fig = go.Figure()

    # Add contour plot for decision boundary (Trace 0)
    fig.add_contour(x=xx[0], y=yy[:, 0], z=Z, opacity=0.1,
                    colorscale='Fall', showlegend=False, showscale=False)

    entity_df = entity_df.groupby("Entity", as_index=False).mean()

    # Add scatter plot for UMAP-transformed data (Trace 1)
    scatter_trace = go.Scatter(x=X['x'], y=X['y'], mode='markers',
                               marker=dict(color=y, colorscale='Fall'),
                               legendgroup='group', showlegend=False)
    fig.add_trace(scatter_trace)

    # Add separate traces for each class for the legend
    classes = set(y)
    for label in classes:
        data_by_label = X[y == label]
        legend_name = '90%-100%' if label == 1 else 'iki 90%'
        entity_texts = entity_df.loc[y == label, 'Entity']
        hover_text = [f'<b>Valstybė:</b> {entity}' for entity in entity_texts]
        fig.add_trace(go.Scatter(x=data_by_label['x'], y=data_by_label['y'], mode='markers',
                                 marker=dict(color=label, colorscale='Fall'),
                                 legendgroup=f'{label}', showlegend=True, name=legend_name,
                                 text=hover_text))

    # Update layout with labels and title
    fig.update_layout(title=title, xaxis_title='UMAP Component 1',
                      yaxis_title='UMAP Component 2')

    fig.show()

'''
# Plot decision boundary for train and test data
plot_decision_boundary(X_train, y_train_encoded, best_classifier,
                       'Apmokymo duomenų pasiskirstymas su normuota duomenų aibe', train_data)
plot_decision_boundary(X_test, y_test_encoded, best_classifier,
                       'Klasifikatoriaus nuspėtų reikšmių iš testavimo duomenų pasiskirstymas su normuota duomenų aibe', test_data)

plot_decision_boundary_umap(X_train_umap, y_train_encoded_umap, best_classifier_umap,
                            'Apmokymo duomenų pasiskirstymas su dvimate dimensija', umap_classification_data)
plot_decision_boundary_umap(X_test_umap, y_test_encoded_umap, best_classifier_umap,
                            'Klasifikatoriaus nuspėtų reikšmių iš testavimo duomenų pasiskirstymas su dvimate dimensija', umap_test_data)


# Confusion matrixes
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix on normalized:\n", conf_matrix)

conf_matrix_umap = confusion_matrix(y_test_umap, y_test_pred_umap)
print("Confusion Matrix on normalized:\n", conf_matrix_umap)
'''


#K-nn starts here
# Prepare data for the original dataset
X_train = train_data[feature_columns]
y_train = train_data['Label']

X_test = test_data[feature_columns]
y_test = test_data['Label']

# Define the hyperparameters you want to tune for k-NN
param_grid = {
    'n_neighbors': [3, 5, 7],  # Different values for the number of neighbors (k)
    'weights': ['uniform', 'distance'],  # Weighting strategy
    'metric': ['euclidean', 'manhattan'],  # Distance metric
}

# Initialize k-NN classifier
knn_classifier = KNeighborsClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)  # 5-fold cross-validation
grid_search.fit(X_train, y_train)

# Get the best parameters found by grid search
best_params = grid_search.best_params_
best_knn_classifier = grid_search.best_estimator_

# Use the best k-NN classifier for final evaluation on test data
y_test_pred = best_knn_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_rep = classification_report(y_test, y_test_pred)

print(f"Best parameters found by grid search: {best_params}")
print(f"Accuracy on test data: {test_accuracy}")
print("Classification Report on test data:\n", test_classification_rep)

# Prepare data for the umap dataset
umap_classification_data['Label'] = (umap_classification_data[TARGET_COLUMN] >= THRESHOLD)
umap_classification_data = umap_classification_data.drop(columns=['Access to electricity (% of population)', 'Entity'], axis=1)

umap_test_data['Label'] = (umap_test_data[TARGET_COLUMN] >= THRESHOLD)
umap_test_data = umap_test_data.drop(columns=['Access to electricity (% of population)', 'Entity'], axis=1)

y_train_umap = umap_classification_data['Label']
X_train_umap = umap_classification_data.drop(columns=['Label'])

y_test_umap = umap_test_data['Label']
X_test_umap = umap_test_data.drop(columns=['Label'])

# Perform grid search with cross-validation for umap dataset
grid_search_umap = GridSearchCV(knn_classifier, param_grid, cv=5)  # 5-fold cross-validation
grid_search_umap.fit(X_train_umap, y_train_umap)

# Get the best parameters found by grid search for umap dataset
best_params_umap = grid_search_umap.best_params_
best_knn_classifier_umap = grid_search_umap.best_estimator_

# Use the best k-NN classifier for final evaluation on test data for umap dataset
y_test_pred_umap = best_knn_classifier_umap.predict(X_test_umap)
test_accuracy_umap = accuracy_score(y_test_umap, y_test_pred_umap)
test_classification_rep_umap = classification_report(y_test_umap, y_test_pred_umap)

print(f"Best parameters found by grid search for umap dataset: {best_params_umap}")
print(f"Accuracy on test data for umap dataset: {test_accuracy_umap}")
print("Classification Report on test data for umap dataset:\n", test_classification_rep_umap)


# Calculate evaluation metrics for the original dataset
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Confusion matrix for the original dataset
conf_matrix = confusion_matrix(y_test, y_test_pred)

print("Original Dataset Metrics:")
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy on test data: {test_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# Calculate evaluation metrics for the UMAP dataset
test_accuracy_umap = accuracy_score(y_test_umap, y_test_pred_umap)
precision_umap = precision_score(y_test_umap, y_test_pred_umap)
recall_umap = recall_score(y_test_umap, y_test_pred_umap)
f1_umap = f1_score(y_test_umap, y_test_pred_umap)

# Confusion matrix for the UMAP dataset
conf_matrix_umap = confusion_matrix(y_test_umap, y_test_pred_umap)

print("\nUMAP Dataset Metrics:")
print("Confusion Matrix:\n", conf_matrix_umap)
print(f"Accuracy on test data (UMAP): {test_accuracy_umap}")
print(f"Precision (UMAP): {precision_umap}")
print(f"Recall (UMAP): {recall_umap}")
print(f"F1 Score (UMAP): {f1_umap}")


# Encode labels for the original dataset
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test_pred)

# Encode labels for the UMAP dataset
y_train_encoded_umap = label_encoder.fit_transform(y_train_umap)
y_test_encoded_umap = label_encoder.transform(y_test_pred_umap)

# Visualize classification results for the original dataset with K-NN
plot_decision_boundary(X_train, y_train_encoded, best_knn_classifier,
                       'Apmokymo duomenų pasiskirstymas su normuota duomenų aibe', train_data)
plot_decision_boundary(X_test, y_test_encoded, best_knn_classifier,
                       'Klasifikatoriaus nuspėtų reikšmių iš testavimo duomenų pasiskirstymas su normuota duomenų aibe', test_data)

# Visualize classification results for the UMAP dataset with K-NN
plot_decision_boundary_umap(X_train_umap, y_train_encoded_umap, best_knn_classifier_umap,
                            'Apmokymo duomenų pasiskirstymas su dvimate dimensija', umap_classification_data)
plot_decision_boundary_umap(X_test_umap, y_test_encoded_umap, best_knn_classifier_umap,
                            'Klasifikatoriaus nuspėtų reikšmių iš testavimo duomenų pasiskirstymas su dvimate dimensija', umap_test_data)
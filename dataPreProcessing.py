import pandas as pd
import plotly.express as px
import numpy as np
import locale
import matplotlib.pyplot as plt


# Set the locale to use thousands separators
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Load the CSV data into object
data = pd.read_csv('Future-500-3.csv')

print(data)

# Extract and convert the numeric part of the "Expenses" column
data['Expenses'] = data['Expenses'].str.replace(
    ' Dollars', '').str.replace(',', '').astype(float)

# Extract and convert the numeric part of the "Revenue" column
data['Revenue'] = data['Revenue'].str.replace(
    '$', '').str.replace(',', '').astype(float)

# Extract and convert the numeric part of the "Growth" column
data['Growth'] = data['Growth'].str.replace(
    '%', '').astype(float)

# Assuming 'data' is your DataFrame
columns_to_exclude = ['ID', 'Name', 'State', 'City',
                      'Industry']  # List of columns to exclude

# Drop the specified columns from the DataFrame
data_filtered = data.drop(columns=columns_to_exclude)


# Calculate quartiles for all numeric columns
quartiles = data_filtered.quantile([0.25, 0.75])

# Display the quartiles for each column
print(f"1st Quartile (25th percentile):\n{quartiles.loc[0.25]}\n")


print(f"3rd Quartile (75th percentile):\n{quartiles.loc[0.75]}\n")


min_values = data_filtered.min()
max_values = data_filtered.max()

print(f"Minimum values:\n {min_values}\n")
print(f"Maximum values:\n {max_values}\n")


print("Average values of columns")
data_mean = data_filtered.mean()
print(data_mean.map(lambda x: "{:.2f}".format(x)))

print("\nMedian values of columns")
data_median = data_filtered.median()
print(data_median)

print("\nStandard deviation values of columns")
data_median = data_filtered.std()
print(data_median.map(lambda x: "{:.2f}".format(x)))

# Group the data by the 'Industry' column
data_grouped_by_industry = data.groupby('Industry')

# Set the display format for float values
pd.options.display.float_format = '{:.2f}'.format

# Iterate through each group (industry) and calculate summary statistics
for industry, group_data in data_grouped_by_industry:

    # Initialize an empty DataFrame to store the summary statistics for the current industry
    industry_summary = pd.DataFrame()

    # Calculate summary statistics for the current industry
    industry_stats = group_data.describe()

    # Add a column to identify the industry in the summary statistics
    industry_stats['Industry'] = industry

    # Calculate the median for numeric columns
    median_row = group_data.select_dtypes(
        include=['number']).median().to_frame().T
    # Rename the index to '50%' to match describe output
    median_row.index = ['median']

    # Concatenate the summary statistics and median for the current industry to the result DataFrame
    industry_summary = pd.concat(
        [industry_summary, industry_stats, median_row])

    # Reset the index of the result DataFrame
    industry_summary.reset_index(inplace=True)

    # Print the summary statistics for the current industry
    print(f"Industry: {industry}")
    print(industry_summary)
    print("\n")


# Create a list to store the indices of rows to remove
rows_to_remove = []

# Iterate through each row
for index, row in data.iterrows():
    # Check if Expenses is null and Revenue and Profit are not null
    if pd.isnull(row['Expenses']) and not pd.isnull(row['Revenue']) and not pd.isnull(row['Profit']):
        # Calculate Expenses based on Revenue and Profit
        expenses = row['Revenue'] - row['Profit']
        # Update the 'Expenses' column with the calculated value
        data.at[index, 'Expenses'] = expenses

    # Check if Revenue is null and Expenses and Profit are not null
    elif pd.isnull(row['Revenue']) and not pd.isnull(row['Expenses']) and not pd.isnull(row['Profit']):
        # Calculate Revenue based on Expenses and Profit
        revenue = row['Profit'] + row['Expenses']
        # Update the 'Revenue' column with the calculated value
        data.at[index, 'Revenue'] = revenue

    # Check if Profit is null and Expenses and Revenue are not null
    elif pd.isnull(row['Profit']) and not pd.isnull(row['Expenses']) and not pd.isnull(row['Revenue']):
        # Calculate Profit based on Revenue and Expenses
        profit = row['Revenue'] - row['Expenses']
        # Update the 'Profit' column with the calculated value
        data.at[index, 'Profit'] = profit

    # Check if Revenue and Expenses and Profit are null
    elif pd.isnull(row['Revenue']) and pd.isnull(row['Profit']) and pd.isnull(row['Expenses']):
        rows_to_remove.append(index)

    # Check if Growth is null
    if pd.isnull(row['Growth']):
        rows_to_remove.append(index)

# Remove the rows outside the loop
data.drop(rows_to_remove, inplace=True)

# Reset the index after removing rows
data.reset_index(drop=True, inplace=True)

# Part 5 Outliers

def remove_outliers(dataset, column_names):
    # Define a multiplier to control the threshold for identifying extreme outliers
    extreme_outlier_multiplier = 3.0
    
    # Create a copy of the dataset to avoid modifying the original data
    data = dataset.copy()
    
    # Iterate through the specified column names
    for column_name in column_names:
        # Calculate the IQR (Interquartile Range) for the current column
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for mild and extreme outliers
        lower_bound_mild = Q1 - 1.5 * IQR
        upper_bound_mild = Q3 + 1.5 * IQR
        lower_bound_extreme = Q1 - extreme_outlier_multiplier * IQR
        upper_bound_extreme = Q3 + extreme_outlier_multiplier * IQR
        
        # Find and count mild outliers
        mild_outliers = data[(data[column_name] < lower_bound_mild) | (data[column_name] > upper_bound_mild)]
        mild_outliers_count = mild_outliers.shape[0]
        
        # Find and count extreme outliers
        extreme_outliers = data[(data[column_name] < lower_bound_extreme) | (data[column_name] > upper_bound_extreme)]
        extreme_outliers_count = extreme_outliers.shape[0]
        
        # Remove outliers from the data
        data = data[~((data[column_name] < lower_bound_mild) | (data[column_name] > upper_bound_mild))]
        
        # Print the counts of mild and extreme outliers for the current column
        print(f"Column: {column_name}")
        print(f"Mild Outliers Count: {mild_outliers_count}")
        print(f"Extreme Outliers Count: {extreme_outliers_count}")
    
    return data

# To test before and after
# fig = px.box(data, y='Revenue')
# fig.show()

data_columns = ["Inception", "Revenue",
                    "Profit", "Expenses", "Employees", "Growth"]
no_outliers_data = remove_outliers(data, data_columns)

# fig = px.box(data, y='Revenue')
# fig.show()

# Part 6 Normalization

# Create a copy of the original DataFrame
normalized_data_min_max = data.copy()
normalized_data_mean_standardization = data.copy()

# List of columns to normalize (numeric columns)
numeric_columns = ['Inception', 'Employees',
                   'Revenue', 'Expenses', 'Profit', 'Growth']

# Min-Max normalization for numeric columns
min_vals = data[numeric_columns].min()
max_vals = data[numeric_columns].max()
normalized_data_min_max[numeric_columns] = (
    data[numeric_columns] - min_vals) / (max_vals - min_vals)

# Mean-Standardization normalization for numeric columns
mean_vals = data[numeric_columns].mean()
std_vals = data[numeric_columns].std()
normalized_data_mean_standardization[numeric_columns] = (
    data[numeric_columns] - mean_vals) / std_vals

# Print the normalized data
print("Original Data:")
print(data)

print("\nNormalized Data Min-Max:")
print(normalized_data_min_max)
print("\nNormalized Data Mean-Standardization:")
print(normalized_data_mean_standardization)

# Part 6 visualization 
# Select the specific column ('Revenue') for plotting
column_to_plot = 'Revenue'

# Extract the data for the selected column in all three versions
original_data_column = data[column_to_plot]
min_max_normalized_column = normalized_data_min_max[column_to_plot]
mean_standardized_column = normalized_data_mean_standardization[column_to_plot]

# Limit the number of values to 50
original_data_column = original_data_column.head(150)
min_max_normalized_column = min_max_normalized_column.head(150)
mean_standardized_column = mean_standardized_column.head(150)

# Create three separate bar charts
plt.figure(figsize=(15, 5))

# Original Data
plt.subplot(1, 3, 1)
plt.bar(range(len(original_data_column)), original_data_column)
plt.title(f'{column_to_plot} (Original)')
plt.xlabel('Index')
plt.ylabel('Value')

# Min-Max Normalized Data
plt.subplot(1, 3, 2)
plt.bar(range(len(min_max_normalized_column)), min_max_normalized_column)
plt.title(f'{column_to_plot} (Min-Max Normalized)')
plt.xlabel('Index')
plt.ylabel('Value')

# Mean-Standardization Normalized Data
plt.subplot(1, 3, 3)
plt.bar(range(len(mean_standardized_column)), mean_standardized_column)
plt.title(f'{column_to_plot} (Mean-Standardization Normalized)')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()


# Part 8 Correlation Matrix
correlation_matrix = data_filtered.corr()
print("Correlation matrix of each numerable field:")
print(correlation_matrix)


# Create a heatmap
plt.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Correlation')

# Set ticks and labels
ticks = np.arange(len(correlation_matrix.columns))
plt.xticks(ticks, correlation_matrix.columns, rotation=45)
plt.yticks(ticks, correlation_matrix.columns)

# Add correlation values to the heatmap
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        if i == j:
            continue
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                 ha='center', va='center', color='#303030', fontsize=8)

# Display the plot
plt.title("Correlation Matrix")
plt.show()


# Part 7 Visual Graphics

# Average growth by industry
print('\nAverage growth by industry')

average_growth_by_industry = data.groupby(
    'Industry')['Growth'].mean().reset_index()
print(average_growth_by_industry)

colors = ['lightblue', 'lightgreen', 'red',
          'purple', 'orange', 'yellow', 'brown']
data_set = []
data_labels = []
for x in average_growth_by_industry['Industry']:
    i = average_growth_by_industry['Industry'].to_list().index(x)
    data_labels.append(x)
    data_set.append([i, average_growth_by_industry['Growth'].to_list()[i]])

fig, ax = plt.subplots()
bplot = ax.boxplot(data_set, patch_artist=True, medianprops={
                   "color": "black", "linewidth": 2})
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel('Industrijos')
ax.set_ylabel('Augimas (%)')
ax.set_xticks([1, 2, 3, 4, 5, 6, 7], data_labels, rotation=45)
ax.set_title('Industrijų augimo procento diagrama')
plt.show()


# Revenue and Profit graph
revenue_data_by_industry = []
for x in data_labels:
    revenue_data_by_industry.append(
        list(data.loc[data['Industry'] == x, 'Revenue']))

profit_data_by_industry = []
for x in data_labels:
    profit_data_by_industry.append(
        list(data.loc[data['Industry'] == x, 'Profit']))

for x in revenue_data_by_industry:
    i = revenue_data_by_industry.index(x)
    plt.scatter(x, profit_data_by_industry[i])

plt.legend(labels=data_labels)
for x in revenue_data_by_industry:
    i = revenue_data_by_industry.index(x)
    plt.plot([sorted(x)[0], sorted(x)[-1]], [sorted(profit_data_by_industry[i])
             [0], sorted(profit_data_by_industry[i])[-1]], marker='o', linewidth=2.5)

plt.xlabel('Pajamos')
plt.ylabel('Pelnas')
plt.title('Pajamų ir pelno priklausomybės industrijai grafikas')
plt.show()


# The number of employees by industry
employees_counts = []
data_copy = data.copy()
rows_to_remove = []
for index, row in data.iterrows():
    if pd.isnull(row['Employees']):
        rows_to_remove.append(index)

data_copy.drop(rows_to_remove, inplace=True)
data_copy.reset_index(drop=True, inplace=True)

for x in data_labels:
    employees_counts.append(
        sum(list(data_copy.loc[data_copy['Industry'] == x, 'Employees'])))

fig, ax = plt.subplots()
bar_container = ax.bar(data_labels, employees_counts, color=colors)
plt.xticks(rotation=45, ha='right')
ax.bar_label(bar_container, fmt='{:,.0f}')
ax.set_ylabel('Darbuotojų skaičius')
ax.set_xlabel('Industrijos')
ax.set_title('Darbuotojų skaičiaus dirbančių industrijoje diagrama')
plt.show()


# Average Revenue by industry
print('\nAverage Revenue by industry')
average_revenue_by_industry = data.groupby(
    'Industry')['Revenue'].mean().reset_index()
print(average_revenue_by_industry)

data_set = []
for x in data_labels:
    i = data_labels.index(x)
    data_set.append(average_revenue_by_industry['Revenue'].to_list()[i])

fig, ax = plt.subplots()
bar_container = ax.bar(data_labels, data_set, color=colors)
plt.xticks(rotation=45, ha='right')
ax.bar_label(bar_container, fmt='{:,.0f}')
ax.set_ylabel('Pajamos ($)')
ax.set_xlabel('Industrijos')
ax.set_title('Vidutinių pajamų industrijai grafikas')
plt.show()


# Expenses by every state of USA
data_copy = data.copy()
rows_to_remove = []
for index, row in data.iterrows():
    if pd.isnull(row['State']):
        rows_to_remove.append(index)

data_copy.drop(rows_to_remove, inplace=True)
data_copy.reset_index(drop=True, inplace=True)

states = list(set(data_copy['State']))
expenses = []
for x in states:
    expenses.append(
        sum(list(data_copy.loc[data_copy['State'] == x, 'Expenses'])))

fig, ax = plt.subplots()
bar_container = ax.bar(states, expenses, color=colors)
ax.set_xticks(states)
ax.bar_label(bar_container, fmt='{:,.0f}')
ax.set_ylabel('Pajamos (100 milijonų)')
ax.set_xlabel('Valstijos')
ax.set_title('Tenkančių pajamų skaičius valstijai grafikas')
plt.show()

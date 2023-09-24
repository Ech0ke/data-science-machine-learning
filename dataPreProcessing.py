import pandas as pd
import plotly.express as px
import numpy as np
import locale
import time
import matplotlib.pyplot as plt


# Set the locale to use thousands separators
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Load the CSV data into object
data = pd.read_csv('Future-500-3.csv')

print(data)


# # Calculate minimum and maximum values for each column
# min_values = data.min()
# max_values = data.max()

# # Display the results
# print("Minimum values:")
# print(min_values)

# print("\nMaximum values:")
# print(max_values)

# print(max_values.dtype)

# min_id = data['ID'].min()
# max_id = data['ID'].max()

# min_name = data['Name'].min()
# max_name = data['Name'].max()


# TODO: fix data type
# min_industry = data['Industry'].min()
# max_industry = data['Industry'].max()

# min_inception = data['Inception'].min()
# max_inception = data['Inception'].max()

# min_employees = data['Employees'].min()
# max_employees = data['Employees'].max()

# TODO: fix data type
# min_state = data['State'].min()
# max_state = data['State'].max()

# TODO: fix data type
# min_city = data['City'].min()
# max_city = data['City'].max()

# min_revenue = data['Revenue'].min()
# max_revenue = data['Revenue'].max()

# min_revenue_formatted = locale.currency(
#     min_revenue, grouping=True, symbol='')
# max_revenue_formatted = locale.currency(
#     max_revenue, grouping=True, symbol='')

# Calculate the minimum and maximum values
# min_expenses = data['Expenses'].min()
# max_expenses = data['Expenses'].max()

# Format the results with commas for thousands and periods for fractions
# min_expenses_formatted = locale.currency(
#     min_expenses, grouping=True, symbol='')
# max_expenses_formatted = locale.currency(
#     max_expenses, grouping=True, symbol='')

# min_profit = data['Profit'].min()
# max_profit = data['Profit'].max()

# min_profit_formatted = locale.currency(
#     min_profit, grouping=True, symbol='')
# max_profit_formatted = locale.currency(
#     max_profit, grouping=True, symbol='')

# Extract and convert the numeric part of the "Expenses" column
data['Expenses'] = data['Expenses'].str.replace(
    ' Dollars', '').str.replace(',', '').astype(float)

# Extract and convert the numeric part of the "Revenue" column
data['Revenue'] = data['Revenue'].str.replace(
    '$', '').str.replace(',', '').astype(float)

# Extract and convert the numeric part of the "Growth" column
data['Growth'] = data['Growth'].str.replace(
    '%', '').astype(float)

# min_growth = data['Growth'].min()
# max_growth = data['Growth'].max()

# Display the formatted results

# print(f"Minimum ID: {min_id}")
# print(f"Maximum ID: {max_id}\n")

# print(f"Minimum name by alphabet: {min_name}")
# print(f"Maximum name by alphabet: {max_name}\n")

# print(f"Minimum industry by alphabet: {min_industry}")
# print(f"Maximum industry by alphabet: {max_industry}")

# print(f"Minimum inception date: {min_inception}")
# print(f"Maximum inception date: {max_inception}\n")

# print(f"Minimum no. of employees: {min_employees:.0f}")
# print(f"Maximum no. of employees: {max_employees:.0f}\n")

# print(f"Minimum state by alphabet: {min_state}")
# print(f"Maximum state by alphabet: {max_state}\n")


# print(f"Minimum city by alphabet: {min_city}")
# print(f"Maximum city by alphabet: {max_city}\n")


# print(f"Minimum Revenue: ${min_revenue_formatted}")
# print(f"Maximum Expenses: ${max_revenue_formatted}\n")

# print(f"Minimum Expenses: {min_expenses_formatted} Dollars")
# print(f"Maximum Expenses: {max_expenses_formatted} Dollars\n")

# print(f"Minimum Profit: {min_profit_formatted}")
# print(f"Maximum Profit: {max_profit_formatted}\n")

# print(f"Minimum growth: {min_growth}")
# print(f"Maximum growth: {max_profit}\n")

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


def remove_outliers(data):
    data_columns = ["Inception", "Revenue",
                    "Profit", "Expenses", "Employees", "Growth"]
    cleaned_data = data.copy()

    for column_name in data_columns:

        # Calculate the first quartile (Q1) and third quartile (Q3)
        q1 = np.percentile(data[column_name], 25)
        q3 = np.percentile(data[column_name], 75)

        # Calculate the interquartile range (IQR)
        iqr = q3 - q1

        # Define the inner and outside barriers
        inner_barrier_lower = q1 - 1.5 * iqr
        inner_barrier_upper = q3 + 1.5 * iqr

        outside_barrier_lower = q1 - 3 * iqr
        outside_barrier_upper = q3 + 3 * iqr

        # Find outliers using the inner and outside barriers
        mild_outliers = []
        extreme_outliers = []
        index = 0
        for value in data[column_name]:
            # Find indexes of mild outliers
            if value < inner_barrier_lower or value > inner_barrier_upper and value > outside_barrier_lower and value < outside_barrier_upper:
                mild_outliers.append(index)
            # Find indeces of extreme outliers
            elif value < outside_barrier_lower or value > outside_barrier_upper:
                extreme_outliers.append(index)
            index += 1

        cleaned_data = cleaned_data.drop(mild_outliers + extreme_outliers)

        # Reset the index after removing rows
        cleaned_data.reset_index(drop=True, inplace=True)

    return cleaned_data

# To test before and after
# fig = px.box(data, y='Revenue')
# fig.show()


no_outliers_data = remove_outliers(data)

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

correlation_matrix = data_filtered.corr()

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

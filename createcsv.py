import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Sample data
data = {
    "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
    "category": ["A", "B", "A", "C", "B", "C"],
    "value": [10, 15, 12, 20, 18, 25],
    "feature1": [5, 6, 7, 8, 9, 10],
    "feature2": [8, 7, 6, 5, 4, 3],
}

#dataframe
df = pd.DataFrame(data)

#saving as csv file
df.to_csv("sample_data.csv", index=False)
print("CSV file 'sample_data.csv' created successfully.")



try:
    data = pd.read_csv('sample_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# Display first few rows
print("First five rows of the dataset:")
print(data.head())

# Inspecting the structure
print("\nDataset info:")
print(data.info())

# Checking for any missing values
print("\nMissing values:")
print(data.isnull().sum())

# Filling or dropping missing values
data_cleaned = data.dropna()
print("\nMissing values after cleaning:")
print(data_cleaned.isnull().sum())

# Compute basic statistics
print("\nBasic Statistics:")
print(data_cleaned.describe())
print("\nBasic Statistics")
print(data_cleaned.describe(include=[np.number]))

# Grouping eg rows by category
if 'category' in data_cleaned.columns:
    numeric_cols = data_cleaned.select_dtypes(include='number').columns
    grouped = data_cleaned.groupby('category')[numeric_cols].mean()
    print("\nMean values by category:")
    print(grouped)

    # Example pattern observation
    interesting = grouped.idxmax()
    print("\nInteresting findings (max values):")
    print(interesting)
else:
    print("No categorical column named 'category' found for grouping.")


# Line Chart (example: trends over time)
if 'date' in data_cleaned.columns and 'value' in data_cleaned.columns:
    data_cleaned['date'] = pd.to_datetime(data_cleaned['date'])
    data_cleaned.sort_values(by='date', inplace=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_cleaned, x='date', y='value')
    plt.title('Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

# Bar Chart
if 'category' in data_cleaned.columns and 'value' in data_cleaned.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data_cleaned, x='category', y='value', estimator='mean')
    plt.title('Comparison of Values by Category')
    plt.xlabel('Category')
    plt.ylabel('Mean Value')
    plt.show()

# Histogram
if 'value' in data_cleaned.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_cleaned, x='value', bins=20, kde=True)
    plt.title('Distribution of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# Scatter Plot
if 'feature1' in data_cleaned.columns and 'feature2' in data_cleaned.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_cleaned, x='feature1', y='feature2', hue='category' if 'category' in data_cleaned.columns else None)
    plt.title('Scatter Plot of Feature1 vs Feature2')
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.legend(title='Category' if 'category' in data_cleaned.columns else None)
    plt.show()
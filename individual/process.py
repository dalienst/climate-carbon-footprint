import pandas as pd

# Load the dataset
data = pd.read_csv("individual/individual_carbonprint.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the column names
print("\nColumn names:")
print(data.columns)

# Check for missing values in each column
missing_values = data.isnull().sum()

print("\nMissing values in each column:")
print(missing_values)

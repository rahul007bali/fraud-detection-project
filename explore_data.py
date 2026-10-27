
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("creditcard.csv")

# Step 2: Display the first 5 transactions
print("FIRST FIVE TRANSACTIONS")
print(df.head())

# Step 3: Check the dataset size
print("\nDATASET SHAPE")
print(df.shape)

# Step 4: Understand the columns and data types
print("\nDATASET INFORMATION")
df.info()

# Step 5: Check for missing values
print("\nMISSING VALUES")
print(df.isnull().sum().sum())

# Step 6: Count legitimate and fraudulent transactions
print("\nTRANSACTION COUNTS")
print(df["Class"].value_counts())

# Step 7: Show fraud percentages
print("\nTRANSACTION PERCENTAGES")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

import pandas as pd

# Load dataset
df = pd.read_csv("Data/nigerian_retail_and_ecommerce_point_of_sale_records.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

print("\nDtypes:")
print(df.dtypes)

# Preview data
print("\nHead:")
print(df.head())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Payment method distribution
print("\nPayment method distribution:")
print(df["payment_method"].value_counts())

# Transaction ID range
print("\nTransaction ID range:", df["transaction_id"].min(), "-", df["transaction_id"].max())

# Store distribution
print("\nTop 5 stores by transaction count:")
print(df["store_name"].value_counts().head())

# City distribution
print("\nCity distribution:")
print(df["city"].value_counts())

# describe
print(df.describe())
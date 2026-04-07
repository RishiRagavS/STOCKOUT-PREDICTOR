import pandas as pd

# Load both files
sales    = pd.read_csv("data/raw/sales_train_evaluation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

print("=" * 50)
print("SALES FILE")
print("=" * 50)
print("Shape:", sales.shape)          # (rows, columns)
print("\nFirst 3 rows:")
print(sales.head(3))
print("\nColumn names (first 20):")
print(sales.columns.tolist()[:20])

print("\n" + "=" * 50)
print("CALENDAR FILE")
print("=" * 50)
print("Shape:", calendar.shape)
print("\nFirst 5 rows:")
print(calendar.head(5))
print("\nAll columns:")
print(calendar.columns.tolist())
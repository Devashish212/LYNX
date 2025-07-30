import pandas as pd

# Load Excel file
df = pd.read_excel("co2_filtered.xlsx", engine='openpyxl')

# Show first few rows
print(df.head())

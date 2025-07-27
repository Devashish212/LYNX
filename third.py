
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate dummy sales data
np.random.seed(42)
products = pd.DataFrame({
    'ProductID': range(1, 6),
    'Name': ['Pen', 'Notebook', 'Pencil', 'Eraser', 'Marker'],
    'Category': ['Stationery']*5,
    'Price': [5, 20, 3, 2, 15]
})

customers = pd.DataFrame({
    'CustomerID': range(1, 6),
    'Name': ['Aarav', 'Dev', 'Sara', 'Ravi', 'Kavya'],
    'Region': ['North', 'South', 'East', 'West', 'Central']
})

sales = pd.DataFrame({
    'SaleID': range(1, 51),
    'ProductID': np.random.choice(products['ProductID'], 50),
    'CustomerID': np.random.choice(customers['CustomerID'], 50),
    'Quantity': np.random.randint(1, 10, 50),
    'SaleDate': [datetime(2023, 1, 1) + timedelta(days=int(x)) for x in np.random.randint(0, 365, 50)]
})

# Save to CSV (to import into Excel / SQL)
products.to_csv('products.csv', index=False)
customers.to_csv('customers.csv', index=False)
sales.to_csv('sales.csv', index=False)
from sklearn.linear_model import LinearRegression

# Group sales monthly
sales['Month'] = sales['SaleDate'].dt.to_period('M').astype(str)
monthly_sales = sales.groupby('Month')['Quantity'].sum().reset_index()

# Forecast future sales
monthly_sales['MonthNum'] = range(len(monthly_sales))
model = LinearRegression()
model.fit(monthly_sales[['MonthNum']], monthly_sales['Quantity'])

next_month = [[len(monthly_sales)]]
prediction = model.predict(next_month)
print(f"Predicted sales next month: {int(prediction[0])}")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score

# Load Excel file
df = pd.read_excel("co2_filtered.xlsx", engine='openpyxl')

le = LabelEncoder()
df['Make1'] = le.fit_transform(df['Make'])

x = df[['Make1']]
y = df[['CO2 Emissions(g/km)']]

x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2, random_state = 1) 

model = LinearRegression()
model.fit(x_train,y_train)

# y=mx+b
print("slope(m):",model.coef_[0])
print('intrecept(b):',model.intercept_)

#predict data
y_pred = model.predict(x_test)
print("predicted:",y_pred)
print("Actual:",y_test.values)

print(mean_squared_error(y_test, y_pred))
print("r2",r2_score(y_test, y_pred))

avg_co2 = df.groupby('Make')['CO2 Emissions(g/km)'].mean().sort_values(ascending=False)

# Plotting
plt.figure(figsize=(15,6))
avg_co2.plot(kind='bar', color='skyblue')

plt.title("Average CO₂ Emissions by Car Make")
plt.xlabel("Car Make (Brand)")
plt.ylabel("Average CO₂ Emissions (g/km)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#save
joblib.dump(model, 'co2_model.pkl')
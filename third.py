# 📦 Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📊 Step 2: Prepare the Data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Marks': [50, 55, 65, 70, 80]
}
df = pd.DataFrame(data)

# 🧠 Step 3: Separate Inputs (X) and Outputs (y)
X = df[['Hours_Studied']]  # input feature (as DataFrame)
y = df['Marks']            # output label (as Series)

# ✂️ Step 4: Split the Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 🛠️ Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Step 6: View What the Model Learned
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# 🔮 Step 7: Predict on Test Data
y_pred = model.predict(X_test)
print("Predicted Marks:", y_pred)
print("Actual Marks   :", y_test.values)

# 📏 Step 8: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score (Accuracy):", r2)

# 📊 Step 9: Plot the Result
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.title('Hours vs Marks - Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

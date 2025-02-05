import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\pavan\Downloads\Civil_Engineering_Regression_Dataset.csv")

X = df[['Building_Height']]
y = df['Construction_Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression Equation: Construction Cost = {slope:.2f} * Building Height + {intercept:.2f}")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test['Building_Height'], y=y_test, label="Actual", color="blue")
sns.lineplot(x=X_test['Building_Height'], y=y_pred, label="Predicted", color="red")
plt.xlabel("Building Height")
plt.ylabel("Construction Cost")
plt.title("Simple Linear Regression: Building Height vs Construction Cost")
plt.legend()
plt.show()

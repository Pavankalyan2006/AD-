import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\pavan\Downloads\Civil_Engineering_Regression_Dataset.csv")

X = df[['Building_Height',  'Labor_Cost', 'Concrete_Strength', 'Foundation_Depth']]
y = df['Construction_Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

coefficients = dict(zip(X.columns, model.coef_))
intercept = model.intercept_
print(f"Regression Equation: Construction Cost = {intercept:.2f} + " + " + ".join([f"{coeff:.2f} * {var}" for var, coeff in coefficients.items()]))

highest_impact_variable = max(coefficients, key=coefficients.get)
print(f"Highest impact variable: {highest_impact_variable} ({coefficients[highest_impact_variable]:.2f})")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.4f}")

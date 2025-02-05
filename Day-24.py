import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv(r"C:\Users\pavan\Downloads\Civil_Engineering_Regression_Dataset.csv")

X = df[['Building_Height', 'Labor_Cost', 'Concrete_Strength', 'Foundation_Depth']]
y = df['Construction_Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression (for comparison)
simple_model = LinearRegression()
simple_model.fit(X_train[['Building_Height']], y_train)
simple_r2 = simple_model.score(X_test[['Building_Height']], y_test)

# Multiple Linear Regression
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
multiple_r2 = multiple_model.score(X_test, y_test)

# Adjusted R-squared Calculation
n = X_test.shape[0]  
p = X_test.shape[1]  
adjusted_r2 = 1 - (1 - multiple_r2) * (n - 1) / (n - p - 1)

# VIF Calculation
X_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# Print Results
print(f"Simple Linear Regression R²: {simple_r2:.4f}")
print(f"Multiple Linear Regression R²: {multiple_r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")
print("\nVariance Inflation Factor (VIF) for detecting multicollinearity:")
print(vif_data)

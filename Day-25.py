import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv(r"C:\Users\pavan\Downloads\Civil_Engineering_Regression_Dataset.csv")

# Define independent and dependent variables
X = df[['Building_Height', 'Labor_Cost', 'Concrete_Strength', 'Foundation_Depth']]
y = df['Construction_Cost']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression (Building Height vs. Construction Cost)
simple_model = LinearRegression()
simple_model.fit(X_train[['Building_Height']], y_train)
simple_r2 = simple_model.score(X_test[['Building_Height']], y_test)
simple_mse = mean_squared_error(y_test, simple_model.predict(X_test[['Building_Height']]))

# Multiple Linear Regression
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
y_pred = multiple_model.predict(X_test)
multiple_r2 = r2_score(y_test, y_pred)
multiple_mse = mean_squared_error(y_test, y_pred)

# Adjusted R-squared Calculation
n, p = X_test.shape  
adjusted_r2 = 1 - (1 - multiple_r2) * (n - 1) / (n - p - 1)

# Regression Coefficients Interpretation
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': multiple_model.coef_})
most_influential_factor = coefficients.loc[coefficients['Coefficient'].abs().idxmax()]

# VIF Calculation for Multicollinearity
X_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# Summary of Findings
print("\nModel Performance:")
print(f"Simple Linear Regression R²: {simple_r2:.4f}, MSE: {simple_mse:.2f}")
print(f"Multiple Linear Regression R²: {multiple_r2:.4f}, Adjusted R²: {adjusted_r2:.4f}, MSE: {multiple_mse:.2f}")
print("\nRegression Coefficients:")
print(coefficients)
print(f"\nMost Influential Factor: {most_influential_factor['Feature']} (Coefficient: {most_influential_factor['Coefficient']:.4f})")
print("\nVariance Inflation Factor (VIF) for Multicollinearity Detection:")
print(vif_data)

# Conclusion
print("\nConclusion:")
print("1. Multiple Regression performs better than Simple Regression for predicting Construction Cost.")
print("2. The most influential factor in cost estimation is:", most_influential_factor['Feature'])
print("3. High VIF values indicate multicollinearity issues that may need feature reduction or transformation.")
print("4. Incorporating additional variables like Geographic Location, Inflation, and Weather Conditions can improve predictions.")
print("5. Regression analysis helps construction companies optimize costs, improve budgeting, and minimize overruns.")

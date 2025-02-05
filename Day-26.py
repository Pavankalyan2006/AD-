import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv(r"C:\Users\pavan\Downloads\Civil_Engineering_Regression_Dataset.csv")

X = df[['Building_Height', 'Labor_Cost', 'Concrete_Strength', 'Foundation_Depth']]
y = df['Construction_Cost']

# Backward Elimination
X = sm.add_constant(X)  # Add constant term for intercept

# Fit the initial model
model = sm.OLS(y, X).fit()
print(model.summary())

# Backward elimination: remove variables with high p-values
while max(model.pvalues) > 0.05:
    excluded_variable = model.pvalues.idxmax()
    X = X.drop(columns=excluded_variable)
    model = sm.OLS(y, X).fit()

print("\nFinal model after backward elimination:")
print(model.summary())

# Lasso Regression (for feature selection)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.drop(columns='const'))  # Scaling the features

lasso = Lasso(alpha=0.1)  # Adjust alpha to control regularization strength
lasso.fit(X_scaled, y)

# Get non-zero coefficients (selected features)
lasso_selected_features = X.columns[lasso.coef_ != 0]
print("\nFeatures selected by Lasso Regression:")
print(lasso_selected_features)

# Residual Analysis
y_pred = model.predict(X)
residuals = y - y_pred

# Plot residuals
plt.figure(figsize=(8,6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Check if residuals are randomly distributed
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Calculate Z-scores for outlier detection
z_scores = np.abs(zscore(X.drop(columns='const')))

# Set a threshold for Z-scores (commonly 3)
outliers = (z_scores > 3).all(axis=1)

# Identify rows with outliers
outlier_data = df[outliers]

print("\nOutliers detected:")
print(outlier_data)

# Impact of outliers on regression model
X_without_outliers = X[~outliers]
y_without_outliers = y[~outliers]

# Refit model without outliers
model_no_outliers = sm.OLS(y_without_outliers, X_without_outliers).fit()
print("\nModel summary without outliers:")
print(model_no_outliers.summary())

# Model Deployment: Suggestions for additional features or real-time data sources
# In a real construction cost estimation tool, consider adding the following:
# - Weather Data (temperature, rainfall)
# - Labor Availability & Costs
# - Supply Chain Data (material availability, prices)
# - Project Scope Changes (adjustments in labor, material)
# - Geographical Factors (terrain, accessibility)

# Ethical Considerations & Decision Making
# - Overestimating Costs: Potential for project abandonment or loss of funding.
# - Underestimating Costs: Financial shortfall, project delays, and compromised safety.
# - Worker Impacts: Incorrect cost predictions may affect labor allocation and wages.
# - Delays: Poor forecasting leads to material shortages and project delays.

# Ensure continuous updating of the model with real-time data to improve accuracy and reduce the risk of errors in predictions.

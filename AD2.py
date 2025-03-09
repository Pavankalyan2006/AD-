# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load Dataset
file_path = r"C:\Users\pavan\OneDrive\Desktop\AD\Ecommerce_customers(1).csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df_cleaned = df.drop(columns=['Email', 'Address', 'Avatar'])

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Splitting Features & Target
X = df_cleaned.drop(columns=['Yearly Amount Spent'])
y = df_cleaned['Yearly Amount Spent']

# Univariate Analysis - Distribution & Outliers
for col in X.columns:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(X[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=X[col], color='red')
    plt.title(f'Boxplot of {col}')

    plt.show()

# Handling Skewness
skewed_features = X.apply(lambda x: skew(x))
skewed_features = skewed_features[abs(skewed_features) > 0.5]  # Selecting highly skewed features

# Apply log transformation
for feature in skewed_features.index:
    X[feature] = np.log1p(X[feature])  # log1p avoids log(0) issues

# Detecting & Treating Outliers using IQR
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

X_no_outliers = X[~((X < lower_bound) | (X > upper_bound)).any(axis=1)]
y_no_outliers = y.loc[X_no_outliers.index]

# Feature Selection - Removing Weak Features
correlation = X_no_outliers.corrwith(y_no_outliers)
low_correlation_features = correlation[abs(correlation) < 0.1].index  # Features with correlation < 0.1

X_final = X_no_outliers.drop(columns=low_correlation_features)
print(f"Dropped features: {list(low_correlation_features)}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_no_outliers, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R² Score": r2}

# Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Training & Evaluating Models
ml_results = []
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    ml_results.append(evaluate_model(y_test, y_pred, model_name))

ml_results_df = pd.DataFrame(ml_results)
print("\n Machine Learning Model Performance:\n", ml_results_df)

# Model Performance Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="R² Score", data=ml_results_df, palette="coolwarm")
plt.title(" Model Comparison - R² Score")
plt.ylim(0, 1)
plt.show()

# Selecting Best Model
best_model_name = ml_results_df.loc[ml_results_df['R² Score'].idxmax(), 'Model']
print(f"\n Best Performing Model: {best_model_name}")

# Hyperparameter Tuning for Best Model
if best_model_name == "Random Forest":
    best_model = RandomForestRegressor()
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
elif best_model_name == "Gradient Boosting":
    best_model = GradientBoostingRegressor()
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}

# Grid Search Optimization
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Train Optimized Model
optimized_model = grid_search.best_estimator_
optimized_model.fit(X_train_scaled, y_train)
final_pred = optimized_model.predict(X_test_scaled)

# Final Evaluation
final_results = evaluate_model(y_test, final_pred, f"Optimized {best_model_name}")
print("\n Final Optimized Model Performance:\n", final_results)

# Feature Importance (Only for Tree-based models)
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    feature_importance = optimized_model.feature_importances_
    importance_df = pd.DataFrame({"Feature": X_final.columns, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
    plt.title(f" Feature Importance - {best_model_name}")
    plt.show()

# Save the Best Model
joblib.dump(optimized_model, "best_model.pkl")
print("\n Model Saved as 'best_model.pkl'")

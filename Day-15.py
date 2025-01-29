import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

df = pd.read_csv(r"C:\Users\pavan\Downloads\Day_15_Healthcare_Data.csv")


print("Initial Data Overview:\n", df.head())
print("\nDataset Info:\n")
df.info()
print("\nMissing Values Per Column:\n", df.isna().sum())

missing_percentage = (df.isna().sum() / len(df)) * 100
print("\nPercentage of Missing Values:\n", missing_percentage)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cmap='viridis', cbar=False)
plt.title("Missing Data Heatmap")
plt.show()

for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

imputer = KNNImputer(n_neighbors=5)
df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

print("\nDataset Statistics After Imputation:\n", df.describe())

plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=[np.number]))
plt.title("Boxplot After Imputation")
plt.xticks(rotation=90)
plt.show()

df.to_csv('cleaned_healthcare_data.csv', index=False)

print("Data Cleaning and Imputation Completed. Cleaned dataset saved as 'cleaned_healthcare_data.csv'")

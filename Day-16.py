import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import os

file_path = r"C:\Users\pavan\Downloads\Day_16_Healthcare_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

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

df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = MinMaxScaler()
df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))

df.to_csv('cleaned_healthcare_data.csv', index=False)

print("Data Cleaning Completed. Cleaned dataset saved as 'cleaned_healthcare_data.csv'")

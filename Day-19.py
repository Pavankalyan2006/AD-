
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv(r"C:\Users\pavan\Downloads\Day 19_E-Commerce_Data.csv")

print("Initial Data Overview:\n", df.head())
df.info()
print("\nMissing Values Per Column:\n", df.isna().sum())

imputer_mean = SimpleImputer(strategy='mean')
df[['Product_Price']] = imputer_mean.fit_transform(df[['Product_Price']])

imputer_mode = SimpleImputer(strategy='most_frequent')
df['Product_Category'] = df['Product_Category'].astype(str)
df['Product_Category'] = imputer_mode.fit_transform(df[['Product_Category']])

df['Purchase_Date'] = df['Purchase_Date'].fillna(method='ffill')

knn_imputer = KNNImputer(n_neighbors=5)
df[['Product_Price']] = knn_imputer.fit_transform(df[['Product_Price']])

df.drop_duplicates(inplace=True)

sns.heatmap(df.isna(), cbar=False, cmap='viridis')
plt.show()

before_stats = df.describe()

scaler = MinMaxScaler()
df[['Product_Price']] = scaler.fit_transform(df[['Product_Price']])

after_stats = df.describe()

print("Summary Statistics Before Imputation:\n", before_stats)
print("\nSummary Statistics After Imputation:\n", after_stats)

df.to_csv('cleaned_ecommerce_data.csv', index=False)
print("Data Cleaning Completed. Cleaned dataset saved as 'cleaned_ecommerce_data.csv'")

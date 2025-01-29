import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from textblob import TextBlob

# Load dataset
df = pd.read_csv(r"C:\Users\pavan\Downloads\Day 19_E-Commerce_Data.csv")

# Identify missing values
print("Initial Data Overview:\n", df.head())
df.info()
print("\nMissing Values Per Column:\n", df.isna().sum())

# Handle missing numerical values
imputer_mean = SimpleImputer(strategy='mean')
df['Customer_Age'] = imputer_mean.fit_transform(df[['Customer_Age']])

df['Rating'] = df['Rating'].clip(1, 5)

# Handle missing textual data
df['Review_Text'] = df['Review_Text'].fillna('No Review')
df['Review_Text'] = df['Review_Text'].apply(lambda x: str(TextBlob(x).correct()))

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize Product_Category spelling inconsistencies
df['Product_Category'] = df['Product_Category'].str.lower().str.strip()

# Identify and handle outliers
sns.boxplot(x=df['Product_Price'])
plt.show()
df['Product_Price'] = np.where(df['Product_Price'] > df['Product_Price'].quantile(0.95), df['Product_Price'].median(), df['Product_Price'])

# Convert categorical data into numerical format
label_encoder = LabelEncoder()
df['Product_Category'] = label_encoder.fit_transform(df['Product_Category'])

# Scale numerical values
scaler = MinMaxScaler()
df[['Product_Price', 'Customer_Age']] = scaler.fit_transform(df[['Product_Price', 'Customer_Age']])

# Save cleaned dataset
df.to_csv('cleaned_ecommerce_reviews.csv', index=False)
print("Data Cleaning Completed. Cleaned dataset saved as 'cleaned_ecommerce_reviews.csv'")

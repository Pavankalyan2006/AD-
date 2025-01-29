import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from textblob import TextBlob

df = pd.read_csv(r"C:\Users\pavan\Downloads\Day_18_Tours_and_Travels.csv")

print("Initial Data Overview:\n", df.head())
df.info()
print("\nMissing Values Per Column:\n", df.isna().sum())

imputer = SimpleImputer(strategy='median')
df['Customer_Age'] = imputer.fit_transform(df[['Customer_Age']])

def fill_missing_text(text):
    return str(TextBlob(text).correct()) if pd.notnull(text) else "No review provided"
df['Review_Text'] = df['Review_Text'].apply(fill_missing_text)

df.drop_duplicates(subset=['Review_Text'], inplace=True)

df['Rating'] = df['Rating'].clip(1, 5)

def correct_spelling(text):
    return str(TextBlob(text).correct()) if pd.notnull(text) else text
df['Tour_Package'] = df['Tour_Package'].apply(correct_spelling)

for col in ['Package_Price', 'Rating']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

label_encoders = {}
for col in ['Tour_Package']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = MinMaxScaler()
df[['Customer_Age', 'Package_Price', 'Rating']] = scaler.fit_transform(df[['Customer_Age', 'Package_Price', 'Rating']])

df.to_csv('cleaned_travel_reviews.csv', index=False)
print("Data Cleaning Completed. Cleaned dataset saved as 'cleaned_travel_reviews.csv'")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day_41 Encoding_Methods_Data.csv")
except FileNotFoundError:
    print("Error: fruit_data.csv not found. Please provide the dataset.")
    exit()

print(df.head())
print(df['Color'].value_counts())
print(df['Size'].value_counts())
print(df['Fruit'].value_counts())

le = LabelEncoder()
df['Color_LabelEncoded'] = le.fit_transform(df['Color'])
df['Size_LabelEncoded'] = le.fit_transform(df['Size'])
df['Fruit_LabelEncoded'] = le.fit_transform(df['Fruit'])

print(df[['Color', 'Color_LabelEncoded']].head())
print(df[['Size', 'Size_LabelEncoded']].head())
print(df[['Fruit', 'Fruit_LabelEncoded']].head())

ohe = OneHotEncoder()
ohe_features = ohe.fit_transform(df[['Color', 'Size', 'Fruit']]).toarray()
ohe_feature_names = ohe.get_feature_names_out(['Color', 'Size', 'Fruit'])
df_ohe = pd.DataFrame(ohe_features, columns=ohe_feature_names)

df_encoded = pd.concat([df, df_ohe], axis=1)

print(df.head())
print(df_encoded.head())

print(f"Original number of features: {len(df.columns)}")
print(f"Number of features after One-Hot Encoding: {len(df_encoded.columns)}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\pavan\Downloads\Civil_Engineering_Regression_Dataset.csv")
print(df.head())

print("Independent Variables:", list(df.columns[:-1]))
print("Dependent Variable:", df.columns[-1])

print(df.isnull().sum())
df.fillna(df.median(), inplace=True)

print(df.describe())

corr_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

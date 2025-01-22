import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:\\Users\\pavan\\Downloads\\Day_14_Pharma_data.csv')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Effectiveness', hue='Product_Name', data=data, ci=None)
plt.title('Average Effectiveness of Drugs Across Regions')
plt.xlabel('Region')
plt.ylabel('Average Effectiveness')
plt.legend(title='Drug')
plt.show()

plt.figure(figsize=(14, 7))
sns.violinplot(x='Product_Name', y='Effectiveness', data=data, inner='quartile', color='skyblue')
plt.title('Distribution of Effectiveness for Each Product')
plt.show()

plt.figure(figsize=(14, 7))
sns.violinplot(x='Product_Name', y='Side_Effects', data=data, inner='quartile', color='lightgreen')
plt.title('Distribution of Side Effects for Each Product')
plt.show()

sns.pairplot(data, vars=['Effectiveness', 'Side_Effects', 'Marketing_Spend'], hue='Product_Name', height=2.5)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Trial_Period', y='Effectiveness', data=data, palette='coolwarm')
plt.title('Effectiveness Across Trial Periods')
plt.xlabel('Trial Period')
plt.ylabel('Effectiveness')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='Marketing_Spend', y='Effectiveness', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Effect of Marketing Spend on Drug Effectiveness')
plt.xlabel('Marketing Spend')
plt.ylabel('Effectiveness')
plt.show()

print(data[['Effectiveness', 'Side_Effects']].corr())

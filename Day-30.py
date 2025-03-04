import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day 30 Market_Basket_Data.csv")
except FileNotFoundError:
    print("Error: market_basket.csv not found. Please provide the dataset.")
    exit()

df['Items'] = df['Items'].str.split(',')
transactions = df['Items'].tolist()

print(df.head())

item_counts = {}
for transaction in transactions:
    for item in transaction:
        if item in item_counts:
            item_counts[item] += 1
        else:
            item_counts[item] = 1

most_frequent_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
print(most_frequent_items[:10])

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules.head())

top_5_confidence = rules.nlargest(5, 'confidence')
print(top_5_confidence)

frequent_pairs = rules[rules['lift'] > 1].sort_values(by='lift', ascending=False)
print(frequent_pairs[['antecedents', 'consequents', 'lift']].head(10))
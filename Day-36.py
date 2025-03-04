import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day_36 Decision_Tree_Data.csv")
except FileNotFoundError:
    print("Error: data.csv not found. Please provide the dataset.")
    exit()

print(df.head())
print(df.describe())
print(df['Target'].value_counts().plot(kind='bar'))
plt.show()

for col in df.columns[:-1]:
    plt.scatter(df[col], df['Target'])
    plt.xlabel(col)
    plt.ylabel('Target')
    plt.show()

X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")
print(f"Decision Tree Precision: {precision_score(y_test, y_pred_dt, average='weighted')}")
print(f"Decision Tree Recall: {recall_score(y_test, y_pred_dt, average='weighted')}")
print(f"Decision Tree F1-score: {f1_score(y_test, y_pred_dt, average='weighted')}")

plt.figure(figsize=(20, 10))
plot_tree(dt_clf, filled=True, feature_names=X.columns)
plt.show()

dt_clf_pruned = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
dt_clf_pruned.fit(X_train, y_train)
y_pred_dt_pruned = dt_clf_pruned.predict(X_test)

print(f"Pruned Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt_pruned)}")

lr_clf = LogisticRegression(random_state=42, max_iter=1000)
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day 45 Feature_Engineering_Data.csv")
except FileNotFoundError:
    print("Error: data.csv not found. Please provide the dataset.")
    exit()

print(df.head())
print(df.describe())

for col in df.columns[:-1]:
    plt.hist(df[col])
    plt.title(col)
    plt.show()

print(df.isnull().sum())

X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
feature_importance_dt = pd.Series(dt_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Decision Tree Feature Importance:\n", feature_importance_dt.head(3))

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
feature_importance_rf = pd.Series(rf_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest Feature Importance:\n", feature_importance_rf.head(3))

least_important_features = feature_importance_rf.tail(2).index.tolist()
X_train_reduced = X_train.drop(least_important_features, axis=1)
X_test_reduced = X_test.drop(least_important_features, axis=1)

rf_clf_reduced = RandomForestClassifier(random_state=42)
rf_clf_reduced.fit(X_train_reduced, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_pred_rf_reduced = rf_clf_reduced.predict(X_test_reduced)

print("Random Forest Metrics (Original Features):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted'))

print("Random Forest Metrics (Reduced Features):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_reduced))
print("Precision:", precision_score(y_test, y_pred_rf_reduced, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rf_reduced, average='weighted'))
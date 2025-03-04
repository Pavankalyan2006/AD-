import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
import pickle
import streamlit as st

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day 49 Decision_Tree_Data.csv")
except FileNotFoundError:
    print("Error: data.csv not found. Please provide the dataset.")
    exit()

print(df.head())
print(df.describe())

for col in df.columns[:-1]:
    plt.hist(df[col])
    plt.title(col)
    plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

print(df.isnull().sum())

X = df.drop('Target', axis=1)
y = df['Target']

mutual_info = pd.Series(mutual_info_classif(X, y), index=X.columns).sort_values(ascending=False)
print("Mutual Information:\n", mutual_info)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

cv_scores = cross_val_score(dt_clf, X, y, cv=5)
print("Cross-Validation Accuracy:", cv_scores.mean())

param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_dt_clf = grid_search.best_estimator_
y_pred_tuned = best_dt_clf.predict(X_test)

print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))

with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(best_dt_clf, file)

# Streamlit app (app.py)

# import streamlit as st
# import pandas as pd
# import pickle

# with open('decision_tree_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# st.title('Decision Tree Prediction App')

# feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6']
# features = {}
# for feature in feature_names:
#     features[feature] = st.number_input(f'Enter {feature}')

# if st.button('Predict'):
#     input_data = pd.DataFrame([features])
#     prediction = model.predict(input_data)
#     probabilities = model.predict_proba(input_data)
#     st.write(f'Prediction: {prediction[0]}')
#     st.write(f'Probability of Class 0: {probabilities[0][0]}')
#     st.write(f'Probability of Class 1: {probabilities[0][1]}')
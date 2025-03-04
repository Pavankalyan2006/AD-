import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day 57 Support Vector Machines.csv")
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

X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("Linear Kernel SVM Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Precision:", precision_score(y_test, y_pred_linear))
print("Recall:", recall_score(y_test, y_pred_linear))
print("F1-Score:", f1_score(y_test, y_pred_linear))

svm_poly = SVC(kernel='poly', random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)

svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("Polynomial Kernel SVM Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_poly))
print("RBF Kernel SVM Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)

print("Best SVM Metrics (Tuned):")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Precision:", precision_score(y_test, y_pred_best))
print("Recall:", recall_score(y_test, y_pred_best))
print("F1-Score:", f1_score(y_test, y_pred_best))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_best))
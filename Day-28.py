import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day_28_Clustering_Dataset.csv")
except FileNotFoundError:
    print("Error: clustering_data.csv not found. Please provide the dataset.")
    exit()

print(df.head())

plt.scatter(df['Feature_1'], df['Feature_2'])
plt.xlabel('Feature_1')
plt.ylabel('Feature_2')
plt.title('Scatter plot of Feature_1 vs. Feature_2')
plt.show()

print(df[['Feature_1', 'Feature_2']].describe())

X = df[['Feature_1', 'Feature_2']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.title('K-Means Clustering (K=3)')
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

true_labels = df['True_Label']
predicted_labels = df['Cluster']

ari = adjusted_rand_score(true_labels, predicted_labels)
print(f"Adjusted Rand Index (ARI): {ari}")

misclassified = df[df['True_Label'] != df['Cluster']]
print("Misclassified Points:")
print(misclassified)

k_values = [2, 4, 5]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f'Cluster_{k}'] = kmeans.fit_predict(X_scaled)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df[f'Cluster_{k}'], cmap='viridis')
    plt.title(f'K-Means Clustering (K={k})')
    plt.show()
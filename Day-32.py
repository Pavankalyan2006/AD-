import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

try:
    df = pd.read_csv(r"C:\Users\pavan\Downloads\Day_32 Recommendation_System_Data.csv")
except FileNotFoundError:
    print("Error: ratings.csv not found. Please provide the dataset.")
    exit()

print(df.head())

top_5_items = df['ItemID'].value_counts().head(5)
print(top_5_items)

average_ratings = df.groupby('ItemID')['Rating'].mean()
print(average_ratings.head())

plt.hist(df['Rating'], bins=5)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

user_item_matrix = df.pivot_table(index='UserID', columns='ItemID', values='Rating').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix_sparse)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_recommendations(user_id, user_item_matrix, user_similarity_df, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    user_ratings = user_item_matrix.loc[user_id]
    recommendations = pd.Series(0, index=user_item_matrix.columns)

    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        recommendations += similar_user_ratings * user_similarity_df[user_id][similar_user]

    recommendations = recommendations.sort_values(ascending=False)
    recommendations = recommendations[user_ratings == 0].head(num_recommendations)
    return recommendations

user_id_to_recommend = df['UserID'].iloc[0]
recommendations = get_recommendations(user_id_to_recommend, user_item_matrix, user_similarity_df)
print(f"Recommendations for User {user_id_to_recommend}:")
print(recommendations)
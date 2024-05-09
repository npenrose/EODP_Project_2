import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

Merged_Data_df = pd.read_csv('Merged-Data.csv')

ratings_matrix = Merged_Data_df.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

def get_similar_users(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(n)
    return similar_users.index.tolist()

def recommend_books(user_id, n=5):
    similar_users = get_similar_users(user_id, n)
    book_recommendations = ratings_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
    already_read = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id] > 0].index
    return book_recommendations.drop(already_read).head(n)

def display_recommendations(user_id, num_recommendations=5):
    recommendations = recommend_books(user_id, num_recommendations)
    print(f"User {user_id} may like the following books based on similar users:")
    for isbn, score in recommendations.items():
        book_title = Merged_Data_df[Merged_Data_df['ISBN'] == isbn]['Book-Title'].values[0]
        print(f"{book_title} - Predicted Rating: {score:.2f}")

user_id = 81
display_recommendations(user_id, 5)

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset for user-based collaborative filtering
Merged_Data_df = pd.read_csv('output files/Merged-Data.csv')

# Create a pivot table for user ratings; fill missing values with 0
ratings_matrix = Merged_Data_df.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Calculate the cosine similarity matrix for users
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# Function to get a list of similar users based on cosine similarity
def get_similar_users(user_id, n=5):
    if user_id in user_similarity_df.index:
        similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).iloc[1:n+1]
        return similar_users.index.tolist()
    else:
        return []

# Function to recommend books based on similar users' ratings
def recommend_books(user_id, n=5):
    if user_id in ratings_matrix.index:
        similar_users = get_similar_users(user_id, n)
        book_recommendations = ratings_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
        already_read = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id] > 0].index
        return book_recommendations.drop(already_read).head(n)
    else:
        return pd.Series()

# Function to display book recommendations for a specific user
def display_recommendations(user_id, num_recommendations=5):
    if user_id in ratings_matrix.index:
        recommendations = recommend_books(user_id, num_recommendations)
        print(f"User {user_id} may like the following books based on similar users:")
        for isbn, score in recommendations.items():
            book_title = Merged_Data_df[Merged_Data_df['ISBN'] == isbn]['Book-Title'].iloc[0]
            print(f"{book_title} - Predicted Rating: {score:.2f}")
    else:
        print(f"User {user_id} does not exist in the dataset.")

# Specify a user ID and display the recommendations
user_id = 258152
display_recommendations(user_id, 5)



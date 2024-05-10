import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Merged-Data.csv')

new_ratings_df = pd.read_csv('data/BX-NewBooksRatings.csv')

filtered_df = df[(df['User-Age'] >= 23) & (df['User-Age'] <= 33)]

# Calculate the number of ratings per author and select only those authors with sufficient data
author_count = filtered_df.groupby('Book-Author').size()
authors_with_enough_data = author_count[author_count > 10].index

# Create a pivot table using data from only the selected authors
filtered_authors_df = filtered_df[filtered_df['Book-Author'].isin(authors_with_enough_data)]
author_ratings = filtered_authors_df.pivot_table(index='User-ID', columns='Book-Author', values='Book-Rating', fill_value=0)

# Calculate cosine similarity
author_similarity = cosine_similarity(author_ratings.T)

# Convert similarity matrix to DataFrame
author_similarity_df = pd.DataFrame(author_similarity, index=author_ratings.columns, columns=author_ratings.columns)

# Recommend books based on the similarity matrix
def recommend_books(user_id, num_recommendations=5):
    if user_id in author_ratings.index:
        user_vector = author_ratings.loc[user_id]
        similar_users = author_similarity_df[user_vector > 0].dot(user_vector)
        similar_users = similar_users.sort_values(ascending=False)
        recommended_books = similar_users.head(num_recommendations)
        return recommended_books.index.tolist()
    else:
        return []

# Evaluate the recommendations with new data
predictions = []
actuals = []
for _, row in new_ratings_df.iterrows():
    predicted_books = recommend_books(row['User-ID'], num_recommendations=1)
    if predicted_books:
        predicted_rating = author_ratings.loc[row['User-ID'], predicted_books[0]] if predicted_books[0] in author_ratings.columns else 0
        predictions.append(predicted_rating)
        actuals.append(row['Book-Rating'])

# Calculate MSE and MAE
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')

# Calculate average similarity
mean_similarity = author_similarity_df.values[np.triu_indices_from(author_similarity_df, k=1)].mean()
print(f"average similarity between age 23 and 33: {mean_similarity:.2f}")

# Show average similarity
plt.hist(author_similarity_df.values[np.triu_indices_from(author_similarity_df, k=1)], bins=20, edgecolor='black')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Similarity among Authors')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('output files/Merged-Data.csv')

filtered_df = df[(df['User-Age'] >= 23) & (df['User-Age'] <= 32)]

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

# Calculate average similarity
mean_similarity = author_similarity_df.values[np.triu_indices_from(author_similarity_df, k=1)].mean()
print(f"average smilarity between age 23 and 32: {mean_similarity:.2f}")

# show average similarity
plt.hist(author_similarity_df.values[np.triu_indices_from(author_similarity_df, k=1)], bins=20, edgecolor='black')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Similarity among Authors')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
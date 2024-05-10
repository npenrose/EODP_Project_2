# Show which author get the highest rated
import pandas as pd

df = pd.read_csv('Merged-Data.csv')

# Create Dataframe between 23 and 33
filtered_df = df[(df['User-Age'] >= 23) & (df['User-Age'] <= 33)]

authors_ratings = filtered_df.groupby('Book-Author')['Book-Rating'].mean()

top_authors = authors_ratings.sort_values(ascending=False)
print(f"The top book authors for the age group 23 to 323 are:\n", top_authors.head())
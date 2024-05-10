import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# finding the relationship between number of ratings and average rating for books
books_by_rating = pd.read_csv('output files/book_ratings_avg.csv')
r = np.corrcoef(books_by_rating['num_ratings'], books_by_rating['avg_rating'])
print(r)

fig = plt.figure()
plt.grid()
plt.scatter(books_by_rating['avg_rating'], books_by_rating['num_ratings'])
plt.ylabel('Number of Ratings')
plt.xlabel('Average Rating')
fig.suptitle('Relationship Between Number of Ratings and Average Rating for Books')
fig.savefig('number_versus_average_ratings.png', format='png')

books_by_rating = books_by_rating.head(500)
books_by_rating = books_by_rating.sort_values(by=['num_ratings'], ascending=False)

books_by_rating.to_csv('top_rated_books.csv')

books_by_rating_2 = pd.read_csv('output files/book_ratings_total.csv')

books_by_rating_2 = books_by_rating_2.head(500)
books_by_rating_2 = books_by_rating_2.sort_values(by='avg_rating', ascending=False).copy()

books_by_rating_2.to_csv('most_rated_books.csv')

# finding the variation in ratings of the top books by the age group
merged_data = pd.read_csv('output files/Merged-Data.csv')
book_stats = merged_data.groupby('Title-Substring')['Book-Rating'].agg(['mean', 'median', 'std', 'var', 'count'])
top_books = book_stats.sort_values(by='count', ascending=False).head(6)

top_books_list = top_books.index.tolist()
top_books_data = merged_data[merged_data['Book-Title'].isin(top_books_list)]

fig = plt.figure(figsize = (20, 14))
sns.boxplot(x='Book-Title', y='Book-Rating', data=top_books_data)
plt.xticks(rotation=45)
plt.xlabel('Book Title')
plt.ylabel('Book Rating')
plt.title('Rating Distribution for Top 5 Most Rated Books by Age Group 23-33')
fig.savefig('top_books_rating_distribution.png')
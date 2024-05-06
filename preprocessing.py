import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def preprocessing():
    books_df = pd.read_csv('data/BX-Books.csv')
    ratings_df = pd.read_csv('data/BX-Ratings.csv')
    users_df = pd.read_csv('data/BX-Users.csv')

    new_books_df = pd.read_csv('data/BX-NewBooks.csv')
    new_books_ratings_df = pd.read_csv('data/BX-NewBooksRatings.csv')
    new_books_users_df = pd.read_csv('data/BX-NewBooksUsers.csv')

    # merge all books, rating, and user data and remove all entries where no user age is specified
    merged_data = pd.merge(ratings_df, books_df, on='ISBN', how='inner')
    merged_data = pd.merge(merged_data, users_df, on='User-ID', how='inner')
    merged_data = merged_data.drop(columns=['ISBN', 'User-Country', 'User-City', 'User-State', 'Book-Publisher'])
    merged_data = merged_data.dropna(subset=['User-Age'])

    print(merged_data)

    # plot a histogram showing the frequency of ratings per age group
    fig = plt.figure()
    plt.hist(merged_data['User-Age'], bins=10)
    plt.xlabel('User Age')
    fig.suptitle('Frequency of Ratings per Age Group')
    fig.savefig('age_and_ratings.png', format='png')

    return

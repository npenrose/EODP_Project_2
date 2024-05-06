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

    merged_data = pd.merge(ratings_df, books_df, on='ISBN', how='inner')
    merged_data = pd.merge(merged_data, users_df, on='User-ID', how='inner')

    merged_data = merged_data.drop(columns=['ISBN', 'User-Country', 'User-City', 'User-State', 'Book-Publisher'])

    print(merged_data)

    return

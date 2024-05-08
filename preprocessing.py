import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binned_statistic

MIN_AGE = 4
MAX_AGE = 100


def age_to_int(age):
    if pd.notna(age):
        int_age = ''.join(filter(str.isdigit, age))
        return int(int_age) if int_age else None
    return None


def convert_special_character(string):
    character_conversions = {'©': 'e', '«': 'e', '¨': 'e',
                             '¼': 'u',
                             '¤': 'a', '£': 'a', '¡': 'a', ' ': 'a',
                             '¶': 'o', '³': 'o', 'µ': 'o', '²': 'o',
                             '­': 'i', '¯': 'i',
                             '§': 'c',
                             '±': 'n'}
    new_string = ''
    corrupted_character = False
    prev_char = ''
    prev_corrupted = ''

    for char in string:
        if char == 'ã' and not corrupted_character:
            corrupted_character = True
            prev_corrupted = char

        elif char in character_conversions.keys():
            new_string += character_conversions[char]
            prev_char = character_conversions[char]
            corrupted_character = False

        elif char == '?' and prev_corrupted != 'ƒ':
            if prev_char == ' ':
                new_string += 'o'
                prev_char = 'o'

            else:
                new_string += 'ss'
                prev_char = 'ss'

            corrupted_character = False

        elif not corrupted_character:
            new_string += char
            prev_char = char

        else:
            prev_corrupted = char

    return new_string


def tokenize_words(string):
    punctuation = ['.', ',', ';', ':', '?', '!', '-', '/', '+', '$', '%', '~']
    unwanted_punctuation = [R'\\', '<', '>', '|']
    words = []
    word = ''
    delimiter = ' '

    string = string.replace('"', '')

    for char in string:
        if char == delimiter:
            if word:
                words.append(word)
                word = ''
        elif char in punctuation:
            if word != '':
                words.append(word)
            words.append(char)
            word = ''
        elif char not in unwanted_punctuation:
            word += char

    if word:
        words.append(word)

    if words:
        new_string = words[0]
        for w in words[1:]:
            new_string += ' ' + w
        return new_string
    else:
        return string


# create dataframes from csv files
books_df = pd.read_csv('data/BX-Books.csv', encoding='windows-1252')
ratings_df = pd.read_csv('data/BX-Ratings.csv')
users_df = pd.read_csv('data/BX-Users.csv')

# convert all age entries to a valid int
users_df['User-Age'] = users_df['User-Age'].apply(age_to_int)

# sort user data by age in descending order
age_sorted_users_df = users_df.sort_values(by=['User-Age'], ascending=False)
age_sorted_users_df.to_csv('User-Ages.csv')

# remove age outliers and state column
users_df = users_df.drop(columns=['User-State'])
users_df = users_df[(users_df['User-Age'] >= MIN_AGE) & (users_df['User-Age'] <= MAX_AGE) | users_df['User-Age'].isna()]

# sort user data by city in descending order
city_and_age_sorted_users_df = users_df.sort_values(by=['User-Country', 'User-City', 'User-Age'], ascending=True)
city_and_age_sorted_users_df.to_csv('User-Cities.csv')

# remove all entries with no listed age
users_df = users_df.dropna(subset=['User-Age'])

# lowercase and sort all book author and title names
books_df = books_df.sort_values(by=['Book-Author', 'Book-Title'], ascending=True)
books_df['Book-Author'] = books_df['Book-Author'].apply(lambda x: x.lower() if isinstance(x, str) else None)
books_df['Book-Title'] = books_df['Book-Title'].apply(lambda x: x.lower() if isinstance(x, str) else None)

# convert weird characters to unicode
books_df['Book-Author'] = books_df['Book-Author'].apply(lambda x: tokenize_words(x))
books_df['Book-Title'] = books_df['Book-Title'].apply(lambda x: tokenize_words(x))

# remove arbitrary punctuation and add space before and after punctuation
books_df['Book-Author'] = books_df['Book-Author'].apply(lambda x: convert_special_character(x) if 'ã' in x else x)
books_df['Book-Title'] = books_df['Book-Title'].apply(lambda x: convert_special_character(x) if 'ã' in x else x)

sorted_books = books_df[['Book-Author', 'Book-Title', 'ISBN']].copy()
sorted_books.to_csv('Books.csv')

# remove anything inside parenthesis

# if there is a colon, check other works by the same author and check for matches before colon


# merge all books, rating, and user data and remove unneeded columns
merged_data = pd.merge(ratings_df, books_df, on='ISBN', how='inner')
merged_data = pd.merge(merged_data, users_df, on='User-ID', how='inner')
merged_data = merged_data.drop(columns=['User-Country', 'User-City', 'Book-Publisher'])

# histogram of age distribution
bins = binned_statistic(users_df['User-Age'], users_df['User-Age'], bins=10, statistic='mean')
fig = plt.figure()
plt.grid()
plt.hist(users_df['User-Age'], bins=bins.bin_edges)
plt.xlabel('User Age')
plt.xticks(bins.bin_edges)
fig.suptitle('Distribution of Users by Age')
fig.savefig('age_distribution.png', format='png')

# histogram of ratings per age group with bin size = 10
bins = binned_statistic(merged_data['User-Age'], merged_data['User-Age'], bins=10, statistic='mean')
fig = plt.figure()
plt.grid()
plt.hist(merged_data['User-Age'], bins=bins.bin_edges)
plt.xlabel('User Age')
plt.xticks(bins.bin_edges)
fig.suptitle('Frequency of Ratings per Age Group')
fig.savefig('age_and_ratings_10.png', format='png')

# histogram of ratings per age group with bin size = 5
bins = binned_statistic(merged_data['User-Age'], merged_data['User-Age'], bins=5, statistic='mean')
fig = plt.figure()
plt.grid()
plt.hist(merged_data['User-Age'], bins=bins.bin_edges)
plt.xlabel('User Age')
plt.xticks(bins.bin_edges)
fig.suptitle('Frequency of Ratings per Age Group')
fig.savefig('age_and_ratings_5.png', format='png')

# create new dataframe of users in the targeted age range
target_users = users_df[(users_df['User-Age'] >= 25) & (users_df['User-Age'] <= 40)]
target_users_ratings = merged_data[(merged_data['User-Age'] >= 25) & (merged_data['User-Age'] <= 40)]

# find the total number of and average ratings for each book
book_ratings = target_users_ratings.groupby('Book-Title').agg(
    num_ratings=pd.NamedAgg(column='Book-Rating', aggfunc='count'),
    avg_rating=pd.NamedAgg(column='Book-Rating', aggfunc='mean')
)

# find the total number of and average ratings for each author
author_ratings = target_users_ratings.groupby('Book-Author').agg(
    num_ratings=pd.NamedAgg(column='Book-Rating', aggfunc='count'),
    avg_rating=pd.NamedAgg(column='Book-Rating', aggfunc='mean')
)
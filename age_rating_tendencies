import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv('Merged-Data.csv')

age_rating = df.groupby('User-Age')['Book-Rating'].mean()

plt.figure(figsize=(10, 6))
plt.plot(age_rating.index, age_rating.values, marker='o', linestyle='-')
plt.xlabel('User Age')
plt.ylabel('Average Book Rating')
plt.title('Average Book Rating by User Age')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

age_rating_data = df[['User-Age', 'Book-Rating']].dropna()
correlation, p_value = pearsonr(age_rating_data['User-Age'], age_rating_data['Book-Rating'])

print(f'Correlation between User Age and Book Rating: {correlation:.2f} (p-value: {p_value:.4f})')

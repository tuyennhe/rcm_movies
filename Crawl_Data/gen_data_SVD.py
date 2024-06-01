import pandas as pd
import numpy as np

num_users = 2000
num_movies = 5230

num_ratings_per_user = int(num_movies * 0.7)

new_user_ids = np.arange(1, num_users + 1)
movie_ids = np.arange(1, num_movies + 1)

probabilities = [0.1, 0.1, 0.2, 0.2, 0.4]
ratings_values = [1, 2, 3, 4, 5]

new_ratings_list = []

for user_id in new_user_ids:
    rated_movies = np.random.choice(movie_ids, size=num_ratings_per_user, replace=False)
    user_ratings = np.random.choice(ratings_values, size=num_ratings_per_user, p=probabilities)
    for movie_id, rating in zip(rated_movies, user_ratings):
        new_ratings_list.append([user_id, movie_id, rating])

new_ratings_df = pd.DataFrame(new_ratings_list, columns=['UserID', 'MovieID', 'Rating'])

new_ratings_df = new_ratings_df.sample(frac=1).reset_index(drop=True)
new_ratings_df.info()

file_path = 'new_ratings_shuffled.csv'
new_ratings_df.to_csv(file_path, index=False)

file_path

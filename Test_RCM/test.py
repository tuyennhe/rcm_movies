from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Thay bằng khóa bí mật của bạn

# Load data
movies = pd.read_csv('./movies.csv', encoding='latin1')
users = pd.read_csv('./users.csv', encoding='latin1')
ratings = pd.read_csv('./ratings.csv', encoding='latin1')

# Check for null values and drop them if necessary
ratings = ratings.dropna(subset=['Rating'])
movies = movies.dropna(subset=['MovieID'])

# Merge ratings with movies
ratings = ratings.merge(movies[['MovieID', 'Title']], on='MovieID')

# Check columns after merge
print(ratings.columns)

# Define a reader for the ratings data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# Split data into trainset and testset
trainset, testset = train_test_split(data, test_size=0.2)

# Build and train the model
model = SVD()
model.fit(trainset)

def recommend_movies(user_id, model, movies, num_recommendations=5):
    # Predict ratings for all movies for the given user
    user_ratings = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movies['MovieID']]
    
    # Sort the movies by predicted rating in descending order
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N recommended movies
    recommended_movie_ids = [movie_id for movie_id, _ in user_ratings[:num_recommendations]]
    recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]
    
    return recommended_movies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_user')
def new_user():
    session['new_ratings'] = []
    return render_template('new_user.html', movies=movies['Title'].tolist(), new_ratings=session['new_ratings'])

@app.route('/add_rating', methods=['POST'])
def add_rating():
    movie_id = int(request.form['movie_id'])
    rating = int(request.form['rating'])
    session['new_ratings'].append((len(users) + 1, movie_id, rating))
    return jsonify(session['new_ratings'])

@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    new_ratings = session.get('new_ratings', [])
    
    # Add new user to users DataFrame
    new_user = {'UserID': len(users) + 1, 'Gender': 'M', 'Age': 25, 'Occupation': 1, 'Zip-code': '00000'}
    users.loc[len(users)] = new_user

    # Add new ratings to ratings DataFrame
    for user_id, movie_id, rating in new_ratings:
        ratings.loc[len(ratings)] = [user_id, movie_id, rating]

    # Merge ratings with movies
    ratings_merged = ratings.merge(movies, on='MovieID')

    # Ensure 'Rating' column exists
    if 'Rating' not in ratings_merged.columns:
        raise ValueError("Rating column not found in merged DataFrame.")

    # Define a reader for the ratings data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_merged[['UserID', 'MovieID', 'Rating']], reader)

    # Split data into trainset and testset
    trainset, testset = train_test_split(data, test_size=0.2)

    # Build and train the model
    model = SVD()
    model.fit(trainset)

    # Recommend movies for the new user
    recommendations = recommend_movies(len(users), model, movies)
    
    # Debugging
    print(recommendations)

    return render_template('recommendations.html', user_id=len(users), recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

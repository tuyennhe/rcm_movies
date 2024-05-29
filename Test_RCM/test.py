from flask import Flask, render_template, request, session, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load data
movies = pd.read_csv('./movies.csv', encoding='latin1')
users = pd.read_csv('./users.csv', encoding='latin1')
ratings = pd.read_csv('./ratings.csv', encoding='latin1')

ratings = ratings.dropna(subset=['Rating'])
movies = movies.dropna(subset=['MovieID'])
movies['Runtime'] = movies['Runtime'].str.replace(' min', '').str.replace(',', '').astype(float).astype('Int64')
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def label_sentiment(description):
    scores = sid.polarity_scores(description)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment

movies['Sentiment'] = movies['Description'].apply(label_sentiment)

def recommend_movie(rating, sentiment):
    if rating > 7 and sentiment == "Positive":
        return "Phim hay nên xem"
    elif rating >= 7.5 and sentiment == "Neutral":
        return "Phim hay"
    elif rating >= 7.5 and sentiment == "Negative":
        return "Phim hay nên xem"
    else:
        return "Không được hay cho lắm"

movies['Recommendation'] = movies.apply(lambda row: recommend_movie(row['Rate'], row['Sentiment']), axis=1)


ratings = ratings.merge(movies[['MovieID', 'Title']], on='MovieID')


# Define a reader for the ratings data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# # Split data into trainset and testset
# trainset, testset = train_test_split(data, test_size=0.2)

# # Build and train the model
# model = SVD()
# model.fit(trainset)

def recommend_movies(user_id, model, movies, num_recommendations=5):
    user_ratings = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movies['MovieID']]

    user_ratings.sort(key=lambda x: x[1], reverse=True)

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
    session['new_ratings'].append((int(ratings['UserID'].max()) + 1, movie_id, rating))
    return jsonify(session['new_ratings'])

@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    new_ratings = session.get('new_ratings', [])

    new_user_id = int(ratings['UserID'].max()) + 1
    new_user = {'UserID': new_user_id, 'Gender': 'M', 'Age': 25, 'Occupation': 1, 'Zip-code': '00000'}
    users.loc[len(users)] = new_user

    for user_id, movie_id, rating in new_ratings:
        ratings.loc[len(ratings)] = [user_id, movie_id, rating]

    ratings_merged = ratings[['UserID', 'MovieID', 'Rating']]

    if 'Rating' not in ratings_merged.columns:
        raise ValueError("Rating column not found in merged DataFrame.")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_merged, reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    # Build and train the model
    model = SVD()
    model.fit(trainset)

    # Recommend movies for the new user
    recommendations = recommend_movies(new_user_id, model, movies)
    
    # Debugging
    print(recommendations)

    return render_template('recommendations.html', user_id=new_user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

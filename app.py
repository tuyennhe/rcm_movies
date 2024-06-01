from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Đọc dữ liệu
movies = pd.read_csv("./Data/Data_final/movies.csv", encoding='latin1')
users = pd.read_csv("./Data/Data_final/users.csv", encoding='latin1')
ratings = pd.read_csv("./Data/Data_final/ratings_small.csv", encoding='latin1')

#Tiền xử lí dữ liệu
ratings = ratings.dropna(subset=['Rating'])
movies = movies.dropna(subset=['MovieID'])
movies = movies.drop_duplicates(subset=['Title'])
movies = movies.reset_index(drop=True)
movies['Year'] = movies['Year'].str.extract(r'(\d{4})')
movies['Year'] = movies['Year'].astype(int)
movies['Runtime'] = movies['Runtime'].str.replace(' min', '').str.replace(',', '').astype(float).astype('Int64')
movies['Genre'] = movies['Genre'].fillna('')
genre_split = movies['Genre'].str.split(', ', expand=True)
genre_split.columns = ['Genre1', 'Genre2', 'Genre3']
movies = pd.concat([movies, genre_split], axis=1)
temp_df = movies['Stars'].str.split(', ', expand=True)
movies['Stars1'] = temp_df[0]
movies['Stars2'] = temp_df[1]
movies['Stars3'] = temp_df[2]
ratings = ratings.merge(movies[['MovieID', 'Title']], on='MovieID')

# Define a reader for the ratings dat
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

def recommend_movies(user_id, model, movies, num_recommendations=5):
    user_ratings = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movies['MovieID']]
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [movie_id for movie_id, _ in user_ratings[:num_recommendations]]
    recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]
    return recommended_movies

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

movies_use = movies[['Title', 'Year', 'Runtime', 'Genre1', 'Genre2', 'Genre3', 'Stars1', 'Stars2', 'Stars3', 'Description', 'Rate', 'Votes', 'Img_link', 'Recommendation']]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_use[['Title','Genre1', 'Genre2', 'Genre3', 'Stars1', 'Stars2', 'Stars3', 'Description']].apply(lambda x: ' '.join(x.dropna()), axis=1))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies_by_title(movie_title, top_n=5):
    idx = movies_use.index[movies_use['Title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    similar_movies_indices = [i[0] for i in sim_scores]
    similar_movies = movies_use.iloc[similar_movies_indices]
    return similar_movies

def recommend_movies_by_text(input_text, top_n=5):
    input_tfidf = tfidf_vectorizer.transform([input_text])
    cosine_scores = linear_kernel(input_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_scores.argsort()[-top_n:][::-1]
    recommended_movies = movies_use.iloc[top_indices]
    return recommended_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_movie = ""
    input_text = ""

    if request.method == 'POST':
        selected_movie = request.form.get('selected_movie', "")
        input_text = request.form.get('input_text', "")

        if selected_movie:
            recommended_movies = recommend_movies_by_title(selected_movie)
            return render_template('index.html', movies=movies_use['Title'],
                                   recommended_movies=recommended_movies.values, selected_movie=selected_movie,
                                   input_text=input_text)
        elif input_text:
            recommended_movies = recommend_movies_by_text(input_text)
            return render_template('index.html', movies=movies_use['Title'],
                                   recommended_movies=recommended_movies.values, selected_movie=selected_movie,
                                   input_text=input_text)

    return render_template('index.html', movies=movies_use['Title'], selected_movie=selected_movie,
                           input_text=input_text)

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('q', '')
    suggestions = movies_use[movies_use['Title'].str.contains(query, case=False, na=False)]['Title'].head(10).tolist()
    return jsonify(suggestions)

@app.route('/new_user', methods=['GET'])
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
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    recommendations = recommend_movies(new_user_id, model, movies)
    
    return render_template('recommendations.html', user_id=new_user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

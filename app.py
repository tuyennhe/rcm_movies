from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Tiền xử lí dữ liệu
movies = pd.read_csv("./Data/Data_final/5232movies_imdb.csv", encoding='latin1')
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
positive_threshold = 7
neutral_threshold_low = 4
neutral_threshold_high = 7

def recommend_movie(rating, sentiment):
    if rating >= positive_threshold or (rating >= neutral_threshold_high and sentiment == "Neutral"):
        return "Hay"
    else:
        return "Khong Hay Cho Lam"
    
movies['Recommendation'] = movies.apply(lambda row: recommend_movie(row['Rating'], row['Sentiment']), axis=1)

movies_use = movies[['Title', 'Year', 'Runtime', 'Genre1', 'Genre2', 'Genre3', 'Stars1', 'Stars2', 'Stars3', 'Description', 'Rating', 'Votes', 'Img_link', 'Recommendation']]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_use[['Genre1', 'Genre2', 'Genre3', 'Stars1', 'Stars2', 'Stars3', 'Description']].apply(lambda x: ' '.join(x.dropna()), axis=1))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(movie_title, top_n=5):
    idx = movies_use.index[movies_use['Title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    similar_movies_indices = [i[0] for i in sim_scores]
    similar_movies = movies_use.iloc[similar_movies_indices]
    return similar_movies

def recommend_movies_based_on_text(input_text, top_n=5):
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
            recommended_movies = recommend_movies(selected_movie)
            return render_template('index.html', movies=movies_use['Title'],
                                   recommended_movies=recommended_movies.values, selected_movie=selected_movie,
                                   input_text=input_text)
        elif input_text:
            recommended_movies = recommend_movies_based_on_text(input_text)
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

if __name__ == '__main__':
    app.run(debug=True)

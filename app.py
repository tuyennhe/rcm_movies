from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack

app = Flask(__name__)

# Đọc dữ liệu từ file CSV
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

# Tạo các cột mới từ các cột tạm thời
movies['Stars1'] = temp_df[0]
movies['Stars2'] = temp_df[1]
movies['Stars3'] = temp_df[2]

movies_use = movies[['Title', 'Year', 'Runtime', 'Genre1', 'Genre2', 'Genre3', 'Stars1', 'Stars2', 'Stars3', 'Description', 'Rating', 'Votes', 'Img_link']]

# Xây dựng hệ thống đề xuất phim
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Tính ma trận TF-IDF cho các yếu tố: thể loại, diễn viên, và nội dung
tfidf_matrix_genre = tfidf_vectorizer.fit_transform(movies_use[['Genre1', 'Genre2', 'Genre3']].apply(lambda x: ' '.join(x.dropna()), axis=1))
tfidf_matrix_stars = tfidf_vectorizer.fit_transform(movies_use[['Stars1', 'Stars2', 'Stars3']].apply(lambda x: ' '.join(x.dropna()), axis=1))
tfidf_matrix_description = tfidf_vectorizer.fit_transform(movies_use['Description'])

# Kết hợp các ma trận thành một
combined_tfidf_matrix = hstack((tfidf_matrix_genre, tfidf_matrix_stars, tfidf_matrix_description))

# Tính toán độ tương đồng giữa các phim
cosine_sim = linear_kernel(combined_tfidf_matrix, combined_tfidf_matrix)

def recommend_movies(movie_title, top_n=5):
    idx = movies_use.index[movies_use['Title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    similar_movies_indices = [i[0] for i in sim_scores]
    similar_movies = movies_use.iloc[similar_movies_indices]
    return similar_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_movie = request.form['selected_movie']
        recommended_movies = recommend_movies(selected_movie)
        return render_template('index.html', movies=movies_use['Title'], recommended_movies=recommended_movies.values, selected_movie=selected_movie)
    else:
        return render_template('index.html', movies=movies_use['Title'])

if __name__ == '__main__':
    app.run(debug=True)

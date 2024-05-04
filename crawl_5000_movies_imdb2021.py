import requests
from bs4 import BeautifulSoup
import pandas as pd

#get full

# Hàm để lấy dữ liệu từ một trang cụ thể
def get_movies_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    scraped_movies = soup.find_all(class_="lister-item mode-detail")

    movies = []

    for movie in scraped_movies:
        title = movie.find("a").text.strip()
        year = movie.find("span", class_="lister-item-year").text.strip("()")

        # Kiểm tra xem phần tử "runtime" có tồn tại không
        runtime_element = movie.find("span", class_="runtime")
        runtime = runtime_element.text.strip() if runtime_element else "N/A"

        rating_element = movie.find("span", class_="ipl-rating-star__rating")
        rating = rating_element.text.strip() if rating_element else "N/A"
        description = movie.find("p", class_="").text.strip()
        director = movie.find("a", href=lambda href: href and "/name/" in href).text.strip()
        stars = [star.text.strip() for star in movie.find_all("a", href=lambda href: href and "/name/" in href)]
        votes_element = movie.find("span", {"name": "nv"})
        votes = votes_element.text.strip() if votes_element else "N/A"

        # Kiểm tra xem thông tin về thể loại có tồn tại không
        genre_element = movie.find("span", class_="genre")
        genre = genre_element.text.strip() if genre_element else "N/A"

        movie_info = {
            "Title": title,
            "Year": year,
            "Runtime": runtime,
            "Genre": genre,
            "Rating": rating,
            "Description": description,
            "Director": director,
            "Stars": ', '.join(stars),
            "Votes": votes
        }

        movies.append(movie_info)

    return movies


# URL của trang IMDb list
base_url = "https://www.imdb.com/list/ls524182345/?ref_=otl_"

# Số trang bạn muốn lấy dữ liệu
num_pages = 1

# List để chứa tất cả các bộ phim từ tất cả các trang
all_movies = []

for page in range(1, num_pages + 1):
    url = f"{base_url}{page}"
    movies_on_page = get_movies_from_page(url)
    all_movies.extend(movies_on_page)

# Tạo DataFrame từ list các bộ phim
df = pd.DataFrame(all_movies)

# Lưu DataFrame vào file Excel
df.to_excel("D:\Code\Do_An_AI\movies_data_2024.xlsx", index=False)

print("Dữ liệu đã được lưu")

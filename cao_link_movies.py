from requests_html import HTMLSession

# Hàm để lấy dữ liệu từ một trang cụ thể
def get_movies_from_page(url):
    session = HTMLSession()
    r = session.get(url)

    # Render JavaScript
    r.html.render(browser_args=["--no-sandbox", "--disable-setuid-sandbox"])

    # Find all movie items
    movie_items = r.html.find('.lister-item.mode-detail')

    movies = []

    for movie in movie_items:
        title = movie.find("a")[1].text.strip()
        year = movie.find("span.lister-item-year")[0].text.strip("()")
        runtime = movie.find("span.runtime")[0].text.strip()
        rating = movie.find("span.ipl-rating-star__rating")[0].text.strip()
        description = movie.find("p")[0].text.strip()
        director = movie.find("a[href^='/name/']")[0].text.strip()
        stars = ", ".join([star.text.strip() for star in movie.find("a[href^='/name/']")[1:]])
        votes = movie.find("span[name='nv']")[0].text.strip()

        movie_info = {
            "Title": title,
            "Year": year,
            "Runtime": runtime,
            "Rating": rating,
            "Description": description,
            "Director": director,
            "Stars": stars,
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
df.to_excel("D:\Code\Do_An_AI\movies_data_2024_test.xlsx", index=False)

print("Dữ liệu đã được lưu")

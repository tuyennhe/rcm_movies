import requests
from bs4 import BeautifulSoup
import pandas as pd


# Hàm để lấy dữ liệu từ một trang
def get_movies_from_page(url):
    # Tạo danh sách để lưu thông tin
    movie_info = []

    # Tải nội dung của trang
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Language': 'en-US,en;q=0.5'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Tìm tất cả các thẻ chứa thông tin phim
    scraped_movies = soup.find_all(class_="lister-item-header")

    # Lặp qua từng phim và thu thập thông tin
    for movie in scraped_movies:
        movie_name = movie.find('a').text.strip()
        movie_year = movie.find('span', class_="lister-item-year text-muted unbold").text.strip()
        movie_info.append((movie_name, movie_year))

    return movie_info


# URL cơ sở của danh sách phim
base_url = "https://www.imdb.com/list/ls524182345/?ref_=otl_"

# Tạo danh sách để lưu thông tin từ tất cả các trang
all_movies = []

# Lặp qua các trang và thu thập dữ liệu
page_number = 1
while True:

    #doan nay down 1 trang thoi, xoa bo if else
    if page_number > 1:
        break
    else:
        url = base_url + str(page_number)
        movies_on_page = get_movies_from_page(url)

        # Nếu không có phim nào trên trang hiện tại, dừng vòng lặp
        if not movies_on_page:
            break

        # Thêm thông tin từ trang hiện tại vào danh sách chung
        all_movies.extend(movies_on_page)
        page_number += 1


# Tạo DataFrame từ danh sách phim
df = pd.DataFrame(all_movies, columns=['Movie Name', 'Year'])

# Lưu DataFrame vào file Excel với encoding UTF-8
df.to_excel("D:\Code\Do_An_AI\movies_name_2024.xlsx", index=False, engine='openpyxl')

print("Dữ liệu đã được lưu")

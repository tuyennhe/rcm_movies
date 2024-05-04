import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_image_info_from_page(url):
    # Tải nội dung của trang
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Language': 'en-US,en;q=0.5'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    scraped_images = soup.find_all("div", class_="lister-item-image ribbonize")

    image_info_list = []

    for img_div in scraped_images:
        img = img_div.find("img", class_="loadlate")
        if img:
            image_src = img.get("loadlate")  # Lấy giá trị của thuộc tính loadlate
            image_alt = img.get("alt")
            image_info_list.append({"Image Alt": image_alt, "Image Src": image_src})

    return image_info_list


# URL của trang IMDb list
base_url = "https://www.imdb.com/list/ls524182345/?ref_=otl_"

# Số trang bạn muốn lấy dữ liệu
num_pages = 1

# List để chứa tất cả các thông tin về ảnh từ tất cả các trang
all_image_info = []

for page in range(1, num_pages + 1):
    url = f"{base_url}{page}"
    image_info_on_page = get_image_info_from_page(url)
    all_image_info.extend(image_info_on_page)

# Tạo DataFrame từ list các thông tin về ảnh
df = pd.DataFrame(all_image_info)

# Lưu DataFrame vào file Excel
df.to_excel("D:\Code\Do_An_AI\img_link_32movies_imdb2024.xlsx", index=False)

print("Dữ liệu đã được lưu")

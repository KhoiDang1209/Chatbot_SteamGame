import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Load app IDs and take the first 50
with open("cleaned_steam_apps.json", "r", encoding="utf-8") as f:
    app_data = json.load(f)
app_ids = [app["appid"] for app in app_data["applist"]["apps"]][:50000]

# Set up a session with retries
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_app_data(app_id):
    try:
        response = session.get(f"https://store.steampowered.com/app/{app_id}/", headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Name
        name_elem = soup.find("div", {"id": "appHubAppName"})
        name = name_elem.text.strip() if name_elem else None

        # Tags (first 3)
        tags = [t.string.strip() for t in soup.find_all("a", {"class": "app_tag"})][:3] if name else []

        # Popular Tags (all)
        popular_tags_div = soup.find("div", {"class": "glance_tags popular_tags"})
        popular_tags = [t.string.strip() for t in popular_tags_div.find_all("a", {"class": "app_tag"})] if popular_tags_div else []

        # User Reviews
        reviews_section = soup.find("div", {"id": "userReviews"})
        all_reviews = None
        all_percent = None
        all_count = None

        if reviews_section:

            all_row = reviews_section.find("div", {"class": "user_reviews_summary_row", "data-tooltip-html": lambda x: x and "last 30 days" not in x})
            if all_row:
                all_reviews = all_row.find("span", {"class": "game_review_summary"}).text.strip() if all_row.find("span", {"class": "game_review_summary"}) else None
                all_tooltip = all_row.get("data-tooltip-html", "")
                if all_tooltip:
                    parts = all_tooltip.split(" of the ")
                    all_percent = parts[0].strip() if parts else None
                    all_count = parts[1].split(" user")[0].replace(",", "") if len(parts) > 1 else None

        # Release Date
        release_elem = soup.find("div", {"class": "date"})
        release_date = release_elem.text.strip() if release_elem else None

        # Developer
        dev_elem = soup.select_one(".details_block b:contains('Developer:')")
        developer = dev_elem.find_next("a").text.strip() if dev_elem else None

        # Publisher
        pub_elem = soup.select_one(".details_block b:contains('Publisher:')")
        publisher = pub_elem.find_next("a").text.strip() if pub_elem else None

        # Price
        price_elem = soup.find("div", {"class": "game_purchase_price"})
        price = price_elem.text.strip() if price_elem else "Free to Play" if soup.find(text="Free to Play") else None

        # Description
        desc_elem = soup.find("div", {"class": "game_description_snippet"})
        description = desc_elem.text.strip() if desc_elem else None

        # Store result
        result = {
            "app_id": app_id,
            "name": name,
            "tags": ",".join(tags),
            "popular_tags": ",".join(popular_tags),
            "all_reviews": all_reviews,
            "all_percent": all_percent,
            "all_count": all_count,
            "release_date": release_date,
            "developer": developer,
            "publisher": publisher,
            "price": price,
            "description": description
        }
        print(f"Success: {app_id} - {name}")
        return result

    except requests.exceptions.Timeout as e:
        print(f"Timeout after retries: {app_id} - {e}")
        return {"app_id": app_id, "name": None, "tags": None, "popular_tags": None, "recent_reviews": None, "recent_percent": None, "recent_count": None, "all_reviews": None, "all_percent": None, "all_count": None, "release_date": None, "developer": None, "publisher": None, "price": None, "description": None}
    except requests.exceptions.RequestException as e:
        print(f"Other error: {app_id} - {e}")
        return {"app_id": app_id, "name": None, "tags": None, "popular_tags": None, "recent_reviews": None, "recent_percent": None, "recent_count": None, "all_reviews": None, "all_percent": None, "all_count": None, "release_date": None, "developer": None, "publisher": None, "price": None, "description": None}

# Use ThreadPoolExecutor with max_workers
data = []
max_workers = 3
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_app = {executor.submit(fetch_app_data, app_id): app_id for app_id in app_ids}
    for future in as_completed(future_to_app):
        app_id = future_to_app[future]
        try:
            result = future.result()
            data.append(result)
        except Exception as e:
            print(f"Exception for app ID {app_id}: {e}")

# Save to CSV with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"steam_game_descriptions_test_{timestamp}.csv"
df = pd.DataFrame(data)
df.to_csv(filename, index=False)
print(f"Saved {len(data)} games to {filename}")
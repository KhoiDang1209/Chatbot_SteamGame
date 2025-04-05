import json
import re

# Load the JSON file
with open("download.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract app list
apps = data["applist"]["apps"]

# Function to check if text contains non-ASCII characters
def is_non_english(text):
    return any(ord(char) > 127 for char in text)  # ASCII only goes up to 127

# Function to check if text contains emojis
def contains_emoji(text):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric shapes
        "\U0001F800-\U0001F8FF"  # Supplemental arrows
        "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # Chess symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return bool(emoji_pattern.search(text))

# Filtering unwanted apps
filtered_apps = [
    app for app in apps
    if app.get("name")  # Has a valid name
    and not any(keyword in app["name"].lower() for keyword in [
        "demo", "trailer", "soundtrack", "wallpaper", "theme", "bundle",
        "collection", "season pass", "upgrade", "dlc", "expansion", "episode",
        "chapter", "mod", "patch", "update", "fix", "guide", "manual",
        "tutorial", "walkthrough", "review", "faq", "interview","vr","edition",
        "#","pack","season","seasons",
        "beta","server","test","tool","tools","test","content","extra","mv","movie",
        "film","series","ost","kit","soundtrack","editor","editors","shader","shaders","render","renders",
        "resource","resources","asset","assets","support","advanced","pro","lite","free","pack","packs",
    ])  # Remove unwanted types
    and not is_non_english(app["name"])  # Remove non-English names
    and not contains_emoji(app["name"])  # Remove emoji-containing names
]

# Remove duplicates by keeping only unique names
seen = set()
unique_apps = []
for app in filtered_apps:
    name_lower = app["name"].lower()
    if name_lower not in seen:
        seen.add(name_lower)
        unique_apps.append(app)

# Sort alphabetically
unique_apps.sort(key=lambda x: x["appid"])

# Save the cleaned data
with open("cleaned_steam_apps.json", "w", encoding="utf-8") as f:
    json.dump({"applist": {"apps": unique_apps}}, f, indent=4)

print(f"Cleaned list saved as cleaned_steam_apps.json with {len(unique_apps)} unique entries.")

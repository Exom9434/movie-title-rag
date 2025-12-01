"""
Collect Korean-English movie title information using TMDB API.
"""

import requests
import pandas as pd
import time
from typing import List, Dict
import os
from dotenv import load_dotenv

# Environment variable setup
load_dotenv()

# TMDB API Key setup
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"


def get_popular_korean_movies(num_pages: int = 5) -> List[Dict]:
    """
    Collect data based on popularity column
    """
    movies = []
    
    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "ko-KR",
            "region": "KR",
            "with_original_language": "ko",  # Korean movies only
            "sort_by": "popularity.desc",
            "page": page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            movies.extend(data.get('results', []))
            print(f"✅ Page {page}/{num_pages} collected ({len(data.get('results', []))} movies)")
            time.sleep(0.5)  # API rate limit safety
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to collect page {page}: {e}")
            continue
    
    return movies


def get_movie_details(movie_id: int) -> Dict:
    """
    Download detailed information for movies in the list (English title included here).
    """
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": API_KEY,
        "language": "en-US"  
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download movie info {movie_id}: {e}")
        return {}


def extract_title_pairs(movies: List[Dict]) -> List[Dict]:
    """
    Construct Korean-English pairs.
    """
    title_pairs = []
    total = len(movies)
    
    for idx, movie in enumerate(movies, 1):
        korean_title = movie.get('title', '')
        movie_id = movie.get('id')
        release_date = movie.get('release_date', '')
        year = release_date[:4] if release_date else ''
        
        # Check English title
        details = get_movie_details(movie_id)
        english_title = details.get('title', '')
        
        # Use only if both titles exist
        if not english_title or not korean_title:
            continue
        
        # Exclude cases where original title is registered in English
        if korean_title == english_title:
            continue
        
        title_pairs.append({
            'movie_id': movie_id,
            'korean_title': korean_title,
            'english_title': english_title,
            'year': year,
            'popularity': movie.get('popularity', 0)
        })
        
        print(f"[{idx}/{total}] {korean_title} → {english_title}")
        time.sleep(0.5) 
    
    return title_pairs


def add_famous_movies() -> List[Dict]:
    """
    Manually add a few movies just in case
    """
    famous_movies = [
        {'movie_id': 496243, 'korean_title': '기생충', 'english_title': 'Parasite', 'year': '2019', 'popularity': 1000},
        {'movie_id': 670, 'korean_title': '올드보이', 'english_title': 'Oldboy', 'year': '2003', 'popularity': 900},
        {'movie_id': 374990, 'korean_title': '부산행', 'english_title': 'Train to Busan', 'year': '2016', 'popularity': 850},
        {'movie_id': 293670, 'korean_title': '아가씨', 'english_title': 'The Handmaiden', 'year': '2016', 'popularity': 800},
        {'movie_id': 27586, 'korean_title': '괴물', 'english_title': 'The Host', 'year': '2006', 'popularity': 750},
        {'movie_id': 411088, 'korean_title': '아이 캔 스피크', 'english_title': 'I Can Speak', 'year': '2017', 'popularity': 700},
        {'movie_id': 18983, 'korean_title': '추격자', 'english_title': 'The Chaser', 'year': '2008', 'popularity': 650},
        {'movie_id': 338967, 'korean_title': '터널', 'english_title': 'Tunnel', 'year': '2016', 'popularity': 600},
        {'movie_id': 46758, 'korean_title': '마더', 'english_title': 'Mother', 'year': '2009', 'popularity': 550},
        {'movie_id': 429203, 'korean_title': '범죄도시', 'english_title': 'The Outlaws', 'year': '2017', 'popularity': 500},
    ]
    return famous_movies


def save_to_csv(title_pairs: List[Dict], filename: str = "movie_titles.csv"):
    """
    Save title pairs to a CSV file.
    """
    df = pd.DataFrame(title_pairs)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['movie_id'], keep='first')
    
    # Sort by popularity
    df = df.sort_values('popularity', ascending=False, ignore_index=True)
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"\nTotal {len(df)} movie titles saved to '{filepath}'.")
    print(f"\nData sample:")
    print(df[['korean_title', 'english_title', 'year']].head(10).to_string(index=False))


def main():
    """Main execution function"""
    print("Starting movie data collection from TMDB API\n")
    
    # Check API key
    if not API_KEY:
        print("❌ TMDB_API_KEY is not set!")
        print("   Please create a .env file and set TMDB_API_KEY.")
        print("   Example: TMDB_API_KEY=your_api_key_here")
        return
    
    # 1. Collect popular Korean movies
    print("1️⃣ Collecting popular Korean movies...")
    movies = get_popular_korean_movies(num_pages=5) 
    print(f"   Movies collected: {len(movies)}\n")
    
    # 2. Extract Korean-English title pairs
    print("2️⃣ Collecting English titles...")
    title_pairs = extract_title_pairs(movies)
    print(f"   Title pairs extracted: {len(title_pairs)}\n")
    
    # 3. Add famous movies
    print("3️⃣ Adding famous movies...")
    famous = add_famous_movies()
    title_pairs.extend(famous)
    print(f"   Movies added: {len(famous)}\n")
    
    # 4. Save to CSV file
    print("4️⃣ Saving to CSV file...")
    save_to_csv(title_pairs)
    
    print("\n✅ Data collection complete!")


if __name__ == "__main__":
    main()
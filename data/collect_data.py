"""
TMDB API를 활용해 한영 영화 제목 정보 수집.
"""

import requests
import pandas as pd
import time
from typing import List, Dict
import os
from dotenv import load_dotenv

# 환경변수 설정
load_dotenv()

# TMDB API Key 설정
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"


def get_popular_korean_movies(num_pages: int = 5) -> List[Dict]:
    """
    popularity 칼럼 기준으로 데이터 수집 진행
    """
    movies = []
    
    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "ko-KR",
            "region": "KR",
            "with_original_language": "ko",  # 한국 영화만 다운
            "sort_by": "popularity.desc",
            "page": page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            movies.extend(data.get('results', []))
            print(f"✅ Page {page}/{num_pages} collected ({len(data.get('results', []))} movies)")
            time.sleep(0.5)  # API Limit 확인 실패 --> 안전하게 
            
        except requests.exceptions.RequestException as e:
            print(f"페이지 수집 실패 {page}: {e}")
            continue
    
    return movies


def get_movie_details(movie_id: int) -> Dict:
    """
    목록에 포함된 영화들의 구체적인 정보 다운로드(영문 타이틀 여기 포함됨).
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
        print(f"영화 정보 다운로드 실패 {movie_id}: {e}")
        return {}


def extract_title_pairs(movies: List[Dict]) -> List[Dict]:
    """
    한-영 쌍 구성.
    """
    title_pairs = []
    total = len(movies)
    
    for idx, movie in enumerate(movies, 1):
        korean_title = movie.get('title', '')
        movie_id = movie.get('id')
        release_date = movie.get('release_date', '')
        year = release_date[:4] if release_date else ''
        
        # 영문 타이틀 체크
        details = get_movie_details(movie_id)
        english_title = details.get('title', '')
        
        # 둘 다 있는 경우만 사용
        if not english_title or not korean_title:
            continue
        
        # 원제가 영어로 등록되어 있는 경우도 제외
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
    혹시 모르니까 몇 개 수동으로 추가
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
    
    # 중복 제거
    df = df.drop_duplicates(subset=['movie_id'], keep='first')
    
    # 정렬
    df = df.sort_values('popularity', ascending=False, ignore_index=True)
    
    # 저장
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"\nTotal {len(df)} movie titles saved to '{filepath}'.")
    print(f"\nData sample:")
    print(df[['korean_title', 'english_title', 'year']].head(10).to_string(index=False))


def main():
    """최종 실행 함수"""
    print("TMDB API에서 영화 데이터 수집 시작\n")
    
    # API 키 확인
    if not API_KEY:
        print("❌ TMDB_API_KEY가 설정되지 않았습니다!")
        print("   .env 파일을 생성하고 TMDB_API_KEY를 설정해주세요.")
        print("   예시: TMDB_API_KEY=your_api_key_here")
        return
    
    # 1. Collect popular Korean movies
    print("1️⃣ Collecting popular Korean movies...")
    movies = get_popular_korean_movies(num_pages=5) 
    print(f"   Movies collected: {len(movies)}\n")
    
    # 2. 한국어-영어 제목 쌍 추출
    print("2. 영어 제목 수집 중...")
    title_pairs = extract_title_pairs(movies)
    print(f"   추출된 제목 쌍: {len(title_pairs)}개\n")
    
    # 3. 유명 영화 추가
    print("3. 유명 영화 추가 중...")
    famous = add_famous_movies()
    title_pairs.extend(famous)
    print(f"   추가된 영화: {len(famous)}편\n")
    
    # 4. CSV 파일로 저장
    print("4. CSV 파일로 저장 중...")
    save_to_csv(title_pairs)
    
    print("\n 데이터 수집 완료!")


if __name__ == "__main__":
    main()
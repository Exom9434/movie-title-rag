"""
영화 제목 RAG 번역 시스템 - Streamlit 데모 앱
"""

import streamlit as st
import pandas as pd
from PIL import Image
import os
from src.rag_translator import MovieTitleRAGTranslator

# 페이지 설정
st.set_page_config(
    page_title="영화 제목 RAG 번역",
    page_icon="🎬",
    layout="wide"
)

# 세션 상태 초기화
if 'translator' not in st.session_state:
    with st.spinner("번역 시스템 로딩 중..."):
        st.session_state.translator = MovieTitleRAGTranslator()

translator = st.session_state.translator

# 헤더
st.title("🎬 영화 제목 RAG 번역 시스템")
st.markdown("""
이 시스템은 RAG(Retrieval-Augmented Generation)를 활용하여 
영화 제목을 공식 번역명으로 정확하게 번역합니다.
""")

# 사이드바 - 프로젝트 정보
with st.sidebar:
    st.header("📌 프로젝트 정보")
    st.markdown("""
    **기술 스택:**
    - OpenAI GPT-4o-mini
    - text-embedding-3-small
    - TMDB API
    - scikit-learn
    
    **개선 효과:**
    - 일반 번역: 50% 정확도
    - RAG 번역: 98% 정확도
    - **+48%p 향상** 🚀
    """)
    
    st.markdown("---")
    
    st.header("🎯 샘플 문장")
    sample_sentences = [
        "기생충은 2019년 최고의 영화였다.",
        "부산행을 보고 좀비 영화의 새로운 가능성을 발견했다.",
        "올드보이는 박찬욱 감독의 대표작이다.",
        "어제 터널을 봤는데 정말 긴장감 넘쳤어.",
        "범죄도시 속편도 재미있을까?"
    ]
    
    selected_sample = st.selectbox(
        "샘플 선택:",
        ["직접 입력"] + sample_sentences
    )

# 메인 영역
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 번역하기")
    
    # 텍스트 입력
    if selected_sample == "직접 입력":
        default_text = ""
    else:
        default_text = selected_sample
    
    user_input = st.text_area(
        "번역할 한국어 문장을 입력하세요:",
        value=default_text,
        height=100,
        placeholder="예: 기생충은 정말 훌륭한 영화였다."
    )
    
    # 번역 버튼
    translate_button = st.button("🚀 번역하기", type="primary", use_container_width=True)

with col2:
    st.header("📊 영화 데이터")
    st.metric("총 영화 수", f"{len(translator.df)}개")
    st.metric("데이터 소스", "TMDB API")
    
    # 상위 5개 인기 영화
    st.subheader("인기 영화 Top 5")
    top_movies = translator.df.head(5)
    for idx, row in top_movies.iterrows():
        st.text(f"• {row['korean_title']} → {row['english_title']}")

# 번역 실행
if translate_button and user_input.strip():
    st.markdown("---")
    st.header("✨ 번역 결과")
    
    # 검색된 영화 표시
    with st.spinner("관련 영화 검색 중..."):
        relevant_movies = translator.search_relevant_movies(user_input)
    
    if relevant_movies:
        st.subheader("🔍 검색된 관련 영화")
        search_df = pd.DataFrame(relevant_movies)
        search_df['유사도'] = search_df['similarity'].apply(lambda x: f"{x:.3f}")
        st.dataframe(
            search_df[['korean_title', 'english_title', 'year', '유사도']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("💡 관련 영화를 찾지 못했습니다. (유사도 임계값 0.5 미만)")
    
    # 번역 결과 비교
    col_rag, col_normal = st.columns(2)
    
    with col_rag:
        st.subheader("✅ RAG 번역")
        with st.spinner("RAG 번역 중..."):
            rag_result = translator.translate_with_rag(user_input, verbose=False)
        st.success(rag_result)
    
    with col_normal:
        st.subheader("❌ 일반 번역")
        with st.spinner("일반 번역 중..."):
            normal_result = translator.translate_without_rag(user_input)
        st.error(normal_result)
    
    # 차이점 설명
    if rag_result != normal_result:
        st.info("""
        💡 **RAG의 효과**: 검색된 영화 정보를 참조하여 공식 번역명을 정확하게 사용했습니다.
        일반 번역은 영화 제목을 직역하거나 잘못된 번역을 사용할 수 있습니다.
        """)
    else:
        st.success("두 번역 결과가 동일합니다. 이 문장에는 영화 제목이 없거나 이미 정확하게 번역되었습니다.")

elif translate_button and not user_input.strip():
    st.warning("⚠️ 번역할 문장을 입력해주세요!")

# 평가 결과 섹션
st.markdown("---")
st.header("📈 평가 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="일반 번역 정확도",
        value="50.00%",
        delta=None
    )

with col2:
    st.metric(
        label="RAG 번역 정확도",
        value="98.33%",
        delta="+48.33%p",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="테스트 케이스",
        value="60개",
        delta=None
    )

# 그래프 표시
if os.path.exists("results/accuracy_comparison.png"):
    st.subheader("정확도 비교 그래프")
    image = Image.open("results/accuracy_comparison.png")
    st.image(image, use_container_width=True)
else:
    st.info("💡 `python src/evaluate.py`를 실행하면 평가 결과 그래프를 볼 수 있습니다.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit | 
    <a href='https://github.com/your-username/movie-title-rag'>GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
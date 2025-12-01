"""
Movie title RAG translation system
Personal project reimplementation of RAG approach developed at the company
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI client configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # Using mini for cost savings (gpt-4o also available)


class MovieTitleRAGTranslator:
    """Movie title RAG translation system"""
    
    def __init__(self, csv_path: str = "data/movie_titles.csv"):
        """
        Args:
            csv_path: Path to movie title database CSV
        """
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.embeddings_cache_path = "data/embeddings_cache.pkl"
        
        self._load_data()
        self._load_or_create_embeddings()
    
    def _load_data(self):
        """Load movie title data from CSV file"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Successfully loaded {len(self.df)} movie titles")
        except FileNotFoundError:
            print(f"❌ Cannot find '{self.csv_path}' file.")
            print("   Please run data/collect_data.py first.")
            raise
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return np.array(response.data[0].embedding)
    
    def _load_or_create_embeddings(self):
        """
        Load embeddings from cache or create new ones
        (Using caching to save API call costs)
        """
        # Load from cache if exists
        if os.path.exists(self.embeddings_cache_path):
            print(f"📦 Loading cached embeddings...")
            with open(self.embeddings_cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.embeddings = cache['embeddings']
                cached_titles = cache['titles']
            
            # Check if cache matches current data
            current_titles = self.df['korean_title'].tolist()
            if cached_titles == current_titles:
                print(f"✅ Using cached embeddings ({len(self.embeddings)} items)")
                return
            else:
                print("⚠️  Data has changed, embeddings need to be regenerated")
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(self.df)} movie titles...")
        print("   (Takes 1-2 minutes on first run, uses cache afterwards)")
        
        embeddings_list = []
        for idx, title in enumerate(self.df['korean_title'], 1):
            embedding = self._get_embedding(title)
            embeddings_list.append(embedding)
            if idx % 20 == 0:
                print(f"   Progress: {idx}/{len(self.df)}")
        
        self.embeddings = np.array(embeddings_list)
        
        # Save cache
        os.makedirs(os.path.dirname(self.embeddings_cache_path), exist_ok=True)
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'titles': self.df['korean_title'].tolist()
            }, f)
        print(f"✅ Embeddings generated and cache saved")
    
    def search_relevant_movies(
        self, 
        query: str, 
        top_k: int = 3, 
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search for movie titles similar to the query
        
        Args:
            query: Search query (text to translate)
            top_k: Return top k results
            threshold: Similarity threshold (exclude below this)
            
        Returns:
            List of retrieved movie information
        """
        # Query embedding
        query_embedding = self._get_embedding(query).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Top k indices
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return only results above threshold
        results = []
        for idx in top_k_indices:
            if similarities[idx] > threshold:
                results.append({
                    'korean_title': self.df.iloc[idx]['korean_title'],
                    'english_title': self.df.iloc[idx]['english_title'],
                    'year': self.df.iloc[idx]['year'],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def translate_with_rag(
        self, 
        text: str, 
        target_language: str = "English",
        verbose: bool = True
    ) -> str:
        """
        Translation using RAG
        
        Args:
            text: Korean text to translate
            target_language: Target language
            verbose: Whether to print search results
            
        Returns:
            Translated text
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"📝 Original: {text}")
            print(f"{'='*60}")
        
        # 1. Search relevant movies
        relevant_movies = self.search_relevant_movies(text)
        
        if verbose:
            print(f"\n🔍 Retrieved relevant movies ({len(relevant_movies)}):")
            if relevant_movies:
                for movie in relevant_movies:
                    print(f"   - {movie['korean_title']} → {movie['english_title']} "
                          f"(similarity: {movie['similarity']:.3f})")
            else:
                print("   (No relevant movies found)")
        
        # 2. Construct context
        if relevant_movies:
            context_lines = []
            for movie in relevant_movies:
                context_lines.append(
                    f"- {movie['korean_title']} → {movie['english_title']}"
                )
            context = "\n".join(context_lines)
        else:
            context = "No relevant movie titles found."
        
        # 3. Construct prompt
        system_prompt = f"""You are a professional Korean to {target_language} translator.

CRITICAL RULES:
1. You MUST strictly follow the movie title translations provided in the Context below.
2. When you encounter a Korean movie title in the text, you MUST use the official English title from the Context.
3. If a movie title is not in the Context, translate it naturally but clearly indicate it's not an official title.
4. Maintain the natural flow and meaning of the original text.
5. Output ONLY the translated text without any explanations or notes.

Context (Official Movie Title Translations):
{context}
"""
        
        user_prompt = f"""Translate the following Korean text to {target_language}:

"{text}"
"""
        
        # 4. Call OpenAI API
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            translation = response.choices[0].message.content.strip()
            
            if verbose:
                print(f"\n✅ Translation result:")
                print(f"   {translation}")
                print(f"{'='*60}\n")
            
            return translation
            
        except Exception as e:
            print(f"❌ Error during translation: {e}")
            return None
    
    def translate_without_rag(
        self, 
        text: str, 
        target_language: str = "English"
    ) -> str:
        """
        Regular translation without RAG (for comparison)
        
        Args:
            text: Korean text to translate
            target_language: Target language
            
        Returns:
            Translated text
        """
        system_prompt = f"""You are a professional Korean to {target_language} translator.
Translate the given text naturally and accurately.
Output ONLY the translated text without any explanations."""
        
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate to {target_language}: {text}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error during translation: {e}")
            return None


def demo():
    """Run demo"""
    print("🎬 Movie Title RAG Translation System Demo\n")
    
    # Check OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is not set!")
        print("   Please add OPENAI_API_KEY to your .env file.")
        return
    
    # Initialize system
    translator = MovieTitleRAGTranslator()
    
    # Test sentences
    test_sentences = [
        "기생충은 2019년 최고의 영화였다.",
        "부산행을 보고 좀비 영화의 새로운 가능성을 발견했다.",
        "올드보이는 박찬욱 감독의 대표작이다.",
        "어제 터널을 봤는데 정말 긴장감 넘쳤어.",
        "범죄도시 속편도 재미있을까?"
    ]
    
    print("\n" + "="*60)
    print("🎯 RAG Translation vs Regular Translation Comparison")
    print("="*60)
    
    for sentence in test_sentences:
        print(f"\n📝 Original: {sentence}")
        print("-" * 60)
        
        # RAG translation
        rag_result = translator.translate_with_rag(sentence, verbose=False)
        print(f"✅ RAG translation: {rag_result}")
        
        # Regular translation
        normal_result = translator.translate_without_rag(sentence)
        print(f"❌ Regular translation: {normal_result}")
        print()


if __name__ == "__main__":
    demo()
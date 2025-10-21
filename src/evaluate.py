"""
Movie title translation accuracy evaluation system
Quantitatively compare performance before and after applying RAG
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from datetime import datetime
from rag_translator import MovieTitleRAGTranslator
import platform

# Automatic Korean font setting by OS
def set_korean_font():
    """Set Korean font based on operating system"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
        print("🍎 macOS detected: Using AppleGothic font")
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
        print("🪟 Windows detected: Using Malgun Gothic font")
    elif system == 'Linux':
        plt.rcParams['font.family'] = 'NanumGothic'
        print("🐧 Linux detected: Using NanumGothic font")
    else:
        print(f"⚠️  Unknown OS ({system}): Using default font")
        # Fallback: use sans-serif
        plt.rcParams['font.family'] = 'sans-serif'
    
    plt.rcParams['axes.unicode_minus'] = False  # Prevent minus sign rendering issues

# Apply font settings
set_korean_font()

class TranslationEvaluator:
    """Translation accuracy evaluation system"""
    
    def __init__(self):
        """Initialize evaluation system"""
        self.translator = MovieTitleRAGTranslator()
        self.test_cases = self._generate_test_cases()
        
    def _generate_test_cases(self) -> List[Dict]:
        """
        Generate test cases
        Various sentences containing movie titles
        
        Returns:
            List of test cases
        """
        # Extract samples from actual movie title data
        sample_movies = self.translator.df.head(20)  # Top 20 movies
        
        test_cases = []
        templates = [
            "{title}은 정말 감동적인 영화였다.",
            "어제 {title}를 봤는데 너무 재미있었어.",
            "{title}은 {year}년 최고의 작품이다.",
            "{title} 감독의 연출이 인상적이었다.",
            "친구가 {title}를 추천해줬어.",
            "{title}의 OST가 정말 좋다.",
            "{title}는 반드시 봐야 할 영화다.",
            "{title} 리뷰를 읽어봤어.",
        ]
        
        for _, movie in sample_movies.iterrows():
            korean_title = movie['korean_title']
            english_title = movie['english_title']
            year = movie['year']
            
            # Generate 2-3 test sentences per movie
            for template in templates[:3]:  # Only first 3 templates
                sentence = template.format(
                    title=korean_title,
                    year=year if year else "2020"
                )
                test_cases.append({
                    'korean_sentence': sentence,
                    'movie_title_korean': korean_title,
                    'movie_title_english': english_title,
                    'year': year
                })
        
        return test_cases
    
    def _check_title_in_translation(
        self, 
        translation: str, 
        expected_title: str
    ) -> bool:
        """
        Check if the correct movie title is included in the translation
        
        Args:
            translation: Translated sentence
            expected_title: Expected English title
            
        Returns:
            True if the title is correctly included
        """
        if not translation:
            return False
        
        # Case-insensitive search
        translation_lower = translation.lower()
        expected_lower = expected_title.lower()
        
        return expected_lower in translation_lower
    
    def evaluate_single_case(
        self, 
        test_case: Dict, 
        use_rag: bool
    ) -> Dict:
        """
        Evaluate a single test case
        
        Args:
            test_case: Test case
            use_rag: Whether to use RAG
            
        Returns:
            Evaluation result
        """
        korean_sentence = test_case['korean_sentence']
        expected_title = test_case['movie_title_english']
        
        # Translate
        if use_rag:
            translation = self.translator.translate_with_rag(
                korean_sentence, 
                verbose=False
            )
        else:
            translation = self.translator.translate_without_rag(
                korean_sentence
            )
        
        # Check accuracy
        is_correct = self._check_title_in_translation(translation, expected_title)
        
        return {
            'korean_sentence': korean_sentence,
            'expected_title': expected_title,
            'translation': translation,
            'is_correct': is_correct,
            'use_rag': use_rag
        }
    
    def run_evaluation(self, num_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full evaluation
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            (RAG results DataFrame, regular translation results DataFrame)
        """
        test_cases = self.test_cases[:num_samples] if num_samples else self.test_cases
        
        print(f"🎯 Starting evaluation: {len(test_cases)} test cases")
        print("="*80)
        
        rag_results = []
        normal_results = []
        
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n[{idx}/{len(test_cases)}] Evaluating...")
            print(f"  Original: {test_case['korean_sentence']}")
            print(f"  Expected: {test_case['movie_title_english']}")
            
            # RAG evaluation
            rag_result = self.evaluate_single_case(test_case, use_rag=True)
            rag_results.append(rag_result)
            print(f"  ✅ RAG: {rag_result['translation']}")
            print(f"     {'✓ Correct' if rag_result['is_correct'] else '✗ Incorrect'}")
            
            # Regular translation evaluation
            normal_result = self.evaluate_single_case(test_case, use_rag=False)
            normal_results.append(normal_result)
            print(f"  ❌ Regular: {normal_result['translation']}")
            print(f"     {'✓ Correct' if normal_result['is_correct'] else '✗ Incorrect'}")
        
        print("\n" + "="*80)
        print("✅ Evaluation complete!")
        
        return pd.DataFrame(rag_results), pd.DataFrame(normal_results)
    
    def calculate_metrics(
        self, 
        rag_df: pd.DataFrame, 
        normal_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            rag_df: RAG results DataFrame
            normal_df: Regular translation results DataFrame
            
        Returns:
            Metrics dictionary
        """
        metrics = {
            'rag_accuracy': (rag_df['is_correct'].sum() / len(rag_df)) * 100,
            'normal_accuracy': (normal_df['is_correct'].sum() / len(normal_df)) * 100,
            'rag_correct_count': rag_df['is_correct'].sum(),
            'normal_correct_count': normal_df['is_correct'].sum(),
            'total_cases': len(rag_df),
            'improvement': 0  # Updated after calculation
        }
        
        metrics['improvement'] = metrics['rag_accuracy'] - metrics['normal_accuracy']
        
        return metrics
    
    def print_summary(self, metrics: Dict):
        """
        Print evaluation summary
        
        Args:
            metrics: Evaluation metrics
        """
        print("\n" + "="*80)
        print("📊 Evaluation Summary")
        print("="*80)
        print(f"\nTotal test cases: {metrics['total_cases']}\n")
        
        print("┌─────────────────────┬──────────┬──────────┬──────────┐")
        print("│ Method              │ Accuracy │ Correct  │ Incorrect│")
        print("├─────────────────────┼──────────┼──────────┼──────────┤")
        print(f"│ RAG Translation     │ {metrics['rag_accuracy']:6.2f}% │ "
              f"{metrics['rag_correct_count']:4d}     │ "
              f"{metrics['total_cases'] - metrics['rag_correct_count']:4d}     │")
        print(f"│ Regular Translation │ {metrics['normal_accuracy']:6.2f}% │ "
              f"{metrics['normal_correct_count']:4d}     │ "
              f"{metrics['total_cases'] - metrics['normal_correct_count']:4d}     │")
        print("└─────────────────────┴──────────┴──────────┴──────────┘")
        
        print(f"\n🚀 Improvement: {metrics['improvement']:+.2f}%p")
        
        if metrics['improvement'] > 0:
            print(f"✅ RAG implementation improved translation accuracy by {metrics['improvement']:.1f}%p!")
        elif metrics['improvement'] == 0:
            print("➖ RAG implementation effect is minimal.")
        else:
            print("⚠️  RAG implementation actually decreased accuracy. System review needed.")
    
    def visualize_results(
        self, 
        metrics: Dict, 
        save_path: str = "results/accuracy_comparison.png"
    ):
        """
        Visualize results
        
        Args:
            metrics: Evaluation metrics
            save_path: Graph save path
        """
        # Create results directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create graphs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Accuracy comparison bar chart
        methods = ['Regular Translation', 'RAG Translation']
        accuracies = [metrics['normal_accuracy'], metrics['rag_accuracy']]
        colors = ['#ff6b6b', '#51cf66']
        
        bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Translation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        
        # Display accuracy above bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Correct/Incorrect count comparison
        correct_counts = [metrics['normal_correct_count'], metrics['rag_correct_count']]
        incorrect_counts = [
            metrics['total_cases'] - metrics['normal_correct_count'],
            metrics['total_cases'] - metrics['rag_correct_count']
        ]
        
        x = range(len(methods))
        width = 0.35
        
        bars1 = ax2.bar([i - width/2 for i in x], correct_counts, width, 
                       label='Correct', color='#51cf66', alpha=0.8, edgecolor='black')
        bars2 = ax2.bar([i + width/2 for i in x], incorrect_counts, width,
                       label='Incorrect', color='#ff6b6b', alpha=0.8, edgecolor='black')
        
        ax2.set_ylabel('Number of Cases', fontsize=12)
        ax2.set_title('Correct/Incorrect Case Count', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Display numbers above bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Graph saved: {save_path}")
        plt.close()
    
    def save_detailed_results(
        self, 
        rag_df: pd.DataFrame, 
        normal_df: pd.DataFrame,
        save_path: str = "results/evaluation_results.csv"
    ):
        """
        Save detailed results to CSV
        
        Args:
            rag_df: RAG results DataFrame
            normal_df: Regular translation results DataFrame
            save_path: CSV save path
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Merge results
        comparison_df = pd.DataFrame({
            'korean_sentence': rag_df['korean_sentence'],
            'expected_title': rag_df['expected_title'],
            'rag_translation': rag_df['translation'],
            'rag_correct': rag_df['is_correct'],
            'normal_translation': normal_df['translation'],
            'normal_correct': normal_df['is_correct'],
            'both_correct': rag_df['is_correct'] & normal_df['is_correct'],
            'only_rag_correct': rag_df['is_correct'] & ~normal_df['is_correct'],
            'only_normal_correct': ~rag_df['is_correct'] & normal_df['is_correct'],
            'both_wrong': ~rag_df['is_correct'] & ~normal_df['is_correct']
        })
        
        comparison_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"💾 Detailed results saved: {save_path}")


def main():
    """Run main evaluation"""
    print("🎬 Movie Title Translation Evaluation System")
    print("="*80)
    
    # Initialize evaluation system
    evaluator = TranslationEvaluator()
    
    # Run evaluation (all or partial)
    print("\nEnter number of samples to evaluate (Enter = all):")
    user_input = input("Number of samples: ").strip()
    num_samples = int(user_input) if user_input else None
    
    rag_df, normal_df = evaluator.run_evaluation(num_samples=num_samples)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(rag_df, normal_df)
    
    # Print results
    evaluator.print_summary(metrics)
    
    # Visualize
    evaluator.visualize_results(metrics)
    
    # Save detailed results
    evaluator.save_detailed_results(rag_df, normal_df)
    
    print("\n" + "="*80)
    print("✅ All evaluations complete!")
    print("   - Graph: results/accuracy_comparison.png")
    print("   - Detailed results: results/evaluation_results.csv")
    print("="*80)


if __name__ == "__main__":
    main()
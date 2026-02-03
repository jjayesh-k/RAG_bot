"""
Complete Working RAG Evaluation Script
=======================================

This script evaluates your phi3.5 RAG application end-to-end.
Connects to your actual FAISS index and Ollama phi3.5 model.

Usage:
    python complete_rag_evaluation.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple, Set
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RAGEvaluator:
    """
    Complete RAG Evaluator for phi3.5
    """
    
    def __init__(
        self,
        chunks_file: str,
        faiss_index_path: str = None,
        test_questions_file: str = "test_questions.json"
    ):
        """
        Initialize evaluator
        
        Args:
            chunks_file: Path to your JSONL chunks file
            faiss_index_path: Path to FAISS index (optional)
            test_questions_file: Path to test questions JSON
        """
        self.chunks_file = chunks_file
        self.faiss_index_path = faiss_index_path
        self.test_questions_file = test_questions_file
        
        # Load chunks
        print("📄 Loading chunks...")
        self.chunks = self.load_chunks()
        print(f"   Loaded {len(self.chunks)} chunks")
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.ollama_available = False
        
        # Try to initialize
        self.initialize_components()
    
    def load_chunks(self) -> List[Dict]:
        """Load chunks from JSONL file"""
        chunks = []
        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        chunks.append(chunk)
            return chunks
        except FileNotFoundError:
            print(f"⚠️  Chunks file not found: {self.chunks_file}")
            return []
    
    def initialize_components(self):
        """Initialize embedding model and FAISS index"""
        
        # Try to load sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            print("🔧 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("   ✅ Embedding model loaded")
        except ImportError:
            print("   ⚠️  sentence-transformers not installed")
            print("      Install: pip install sentence-transformers")
        
        # Try to load FAISS
        try:
            import faiss
            if self.faiss_index_path and Path(self.faiss_index_path).exists():
                print("🔧 Loading FAISS index...")
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                print(f"   ✅ FAISS index loaded ({self.faiss_index.ntotal} vectors)")
        except ImportError:
            print("   ⚠️  FAISS not installed")
            print("      Install: pip install faiss-cpu")
        except Exception as e:
            print(f"   ⚠️  Could not load FAISS index: {e}")
        
        # Check Ollama
        try:
            import ollama
            # Test connection
            ollama.list()
            self.ollama_available = True
            print("✅ Ollama available")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("   Make sure Ollama is running: ollama serve")
    
    def retrieve_chunks_simple(self, question: str, top_k: int = 5) -> List[str]:
        """
        Simple retrieval using cosine similarity
        Falls back to keyword matching if embeddings not available
        """
        if self.embedding_model is None:
            # Fallback: keyword matching
            return self.retrieve_chunks_keyword(question, top_k)
        
        # Embed question
        question_embedding = self.embedding_model.encode([question])
        
        # Embed all chunks if not using FAISS
        if self.faiss_index is None:
            chunk_texts = [c['content'] for c in self.chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            # Compute similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            
            # Get top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
        else:
            # Use FAISS
            distances, indices = self.faiss_index.search(
                question_embedding.astype('float32'), 
                top_k
            )
            top_indices = indices[0]
        
        # Return chunk contents
        retrieved = [self.chunks[idx]['content'] for idx in top_indices if idx < len(self.chunks)]
        return retrieved
    
    def retrieve_chunks_keyword(self, question: str, top_k: int = 5) -> List[str]:
        """Fallback: simple keyword-based retrieval"""
        question_words = set(question.lower().split())
        
        # Score each chunk
        scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk['content'].lower().split())
            overlap = len(question_words.intersection(chunk_words))
            scores.append(overlap)
        
        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved = [self.chunks[idx]['content'] for idx in top_indices]
        
        return retrieved
    
    def generate_answer_ollama(self, question: str, contexts: List[str]) -> str:
        """Generate answer using Ollama phi3.5"""
        
        if not self.ollama_available:
            return "Ollama not available - placeholder answer"
        
        # Create prompt
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided contexts. If the answer is not in the contexts, say "I cannot answer based on the provided information."

{context_text}

Question: {question}

Answer:"""
        
        try:
            import ollama
            
            response = ollama.chat(
                model='phi3.5:3.8b-mini-instruct-q4_K_M',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.1,  # Low temperature for factual responses
                    'num_predict': 256   # Reasonable length
                }
            )
            
            answer = response['message']['content']
            return answer.strip()
            
        except Exception as e:
            print(f"⚠️  Error generating answer: {e}")
            return "Error generating answer"
    
    def query_rag(self, question: str, top_k: int = 5) -> Tuple[str, List[str]]:
        """
        Complete RAG pipeline
        
        Returns:
            Tuple of (answer, contexts)
        """
        # Retrieve
        contexts = self.retrieve_chunks_simple(question, top_k)
        
        # Generate
        answer = self.generate_answer_ollama(question, contexts)
        
        return answer, contexts
    
    # ========================================================================
    # Evaluation Metrics
    # ========================================================================
    
    def tokenize(self, text: str) -> Set[str]:
        """Tokenize and normalize text"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        tokens = [t for t in tokens if t not in stop_words]
        return set(tokens)
    
    def calculate_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """How much of ground truth is in retrieved contexts"""
        gt_tokens = self.tokenize(ground_truth)
        context_tokens = self.tokenize(' '.join(contexts))
        
        if len(gt_tokens) == 0:
            return 1.0
        
        overlap = gt_tokens.intersection(context_tokens)
        return len(overlap) / len(gt_tokens)
    
    def calculate_context_precision(self, question: str, contexts: List[str], ground_truth: str) -> float:
        """How relevant are retrieved contexts"""
        question_tokens = self.tokenize(question)
        gt_tokens = self.tokenize(ground_truth)
        relevant_tokens = question_tokens.union(gt_tokens)
        
        if not contexts:
            return 0.0
        
        precision_scores = []
        for context in contexts:
            context_tokens = self.tokenize(context)
            if len(context_tokens) == 0:
                precision_scores.append(0.0)
            else:
                overlap = context_tokens.intersection(relevant_tokens)
                precision_scores.append(len(overlap) / len(context_tokens))
        
        return np.mean(precision_scores)
    
    def calculate_answer_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Is answer supported by contexts"""
        answer_tokens = self.tokenize(answer)
        context_tokens = self.tokenize(' '.join(contexts))
        
        if len(answer_tokens) == 0:
            return 1.0
        
        supported = answer_tokens.intersection(context_tokens)
        return len(supported) / len(answer_tokens)
    
    def calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """Does answer address the question"""
        question_tokens = self.tokenize(question)
        answer_tokens = self.tokenize(answer)
        
        if len(question_tokens) == 0:
            return 1.0
        
        overlap = question_tokens.intersection(answer_tokens)
        return len(overlap) / len(question_tokens)
    
    def evaluate_single_query(
        self,
        question: str,
        ground_truth: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, float]:
        """Evaluate a single query"""
        
        return {
            'context_recall': self.calculate_context_recall(ground_truth, contexts),
            'context_precision': self.calculate_context_precision(question, contexts, ground_truth),
            'answer_faithfulness': self.calculate_answer_faithfulness(answer, contexts),
            'answer_relevancy': self.calculate_answer_relevancy(question, answer),
            'num_contexts': len(contexts),
            'answer_length': len(answer.split())
        }
    
    # ========================================================================
    # Main Evaluation
    # ========================================================================
    
    def run_evaluation(self, max_questions: int = None) -> pd.DataFrame:
        """Run complete evaluation"""
        
        print("\n" + "="*70)
        print("RAG EVALUATION")
        print("="*70)
        
        # Load test questions
        with open(self.test_questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_questions = data['test_questions']
        
        if max_questions:
            test_questions = test_questions[:max_questions]
        
        print(f"\n📊 Evaluating {len(test_questions)} questions...")
        
        results = []
        
        for i, item in enumerate(test_questions, 1):
            print(f"\n{'─'*70}")
            print(f"Question {i}/{len(test_questions)}")
            print(f"{'─'*70}")
            
            question = item['question']
            ground_truth = item['ground_truth']
            
            print(f"Q: {question}")
            
            # Query RAG
            try:
                answer, contexts = self.query_rag(question, top_k=5)
                print(f"A: {answer[:100]}..." if len(answer) > 100 else f"A: {answer}")
                print(f"Retrieved: {len(contexts)} contexts")
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
            
            # Evaluate
            metrics = self.evaluate_single_query(question, ground_truth, answer, contexts)
            
            # Print metrics
            print(f"\nMetrics:")
            print(f"  Recall:       {metrics['context_recall']:.3f}")
            print(f"  Precision:    {metrics['context_precision']:.3f}")
            print(f"  Faithfulness: {metrics['answer_faithfulness']:.3f}")
            print(f"  Relevancy:    {metrics['answer_relevancy']:.3f}")
            
            # Store result
            result = {
                'id': item['id'],
                'category': item['category'],
                'difficulty': item['difficulty'],
                'question': question,
                'ground_truth': ground_truth,
                'generated_answer': answer,
                **metrics
            }
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Display summary
        self.display_summary(df)
        
        return df
    
    def display_summary(self, df: pd.DataFrame):
        """Display evaluation summary"""
        
        print("\n\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        # Overall metrics
        print("\n📊 Overall Metrics:")
        print("─"*70)
        
        metrics = ['context_recall', 'context_precision', 'answer_faithfulness', 'answer_relevancy']
        
        for metric in metrics:
            avg = df[metric].mean()
            std = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            # Status
            if metric == 'context_recall':
                status = "✅" if avg >= 0.65 else "⚠️" if avg >= 0.50 else "❌"
            elif metric == 'context_precision':
                status = "✅" if avg >= 0.70 else "⚠️" if avg >= 0.55 else "❌"
            elif metric == 'answer_faithfulness':
                status = "✅" if avg >= 0.75 else "⚠️" if avg >= 0.60 else "❌"
            else:  # relevancy
                status = "✅" if avg >= 0.65 else "⚠️" if avg >= 0.50 else "❌"
            
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  {status} Average: {avg:.4f} (±{std:.4f})")
            print(f"     Range: [{min_val:.4f}, {max_val:.4f}]")
        
        # By difficulty
        print("\n\n📈 Performance by Difficulty:")
        print("─"*70)
        
        for difficulty in ['easy', 'medium', 'hard']:
            subset = df[df['difficulty'] == difficulty]
            if len(subset) > 0:
                print(f"\n{difficulty.upper()} ({len(subset)} questions):")
                print(f"  Recall:    {subset['context_recall'].mean():.4f}")
                print(f"  Precision: {subset['context_precision'].mean():.4f}")
        
        # Weak areas
        print("\n\n⚠️  Questions Needing Attention:")
        print("─"*70)
        
        weak = df[(df['context_recall'] < 0.5) | (df['context_precision'] < 0.5)]
        
        if len(weak) > 0:
            for _, row in weak.head(5).iterrows():
                print(f"\nQ{row['id']}: {row['question'][:60]}...")
                print(f"  Recall: {row['context_recall']:.3f}, Precision: {row['context_precision']:.3f}")
        else:
            print("  None! All questions performed well ✅")
        
        print("\n" + "="*70)
    
    def save_results(self, df: pd.DataFrame):
        """Save results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV
        csv_file = f"rag_evaluation_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file}")
        
        # Summary report
        report_file = f"rag_evaluation_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAG EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("─"*70 + "\n")
            f.write(f"Total Questions: {len(df)}\n")
            f.write(f"Context Recall:       {df['context_recall'].mean():.4f}\n")
            f.write(f"Context Precision:    {df['context_precision'].mean():.4f}\n")
            f.write(f"Answer Faithfulness:  {df['answer_faithfulness'].mean():.4f}\n")
            f.write(f"Answer Relevancy:     {df['answer_relevancy'].mean():.4f}\n")
        
        print(f"📄 Report saved to: {report_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           COMPLETE RAG EVALUATION SCRIPT                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    CHUNKS_FILE = "knowledge_base_chunks.jsonl"  # ← CHANGE THIS to your chunks file
    FAISS_INDEX = None  # Optional: "your_index.faiss"
    TEST_QUESTIONS = "ground_truth.json"
    
    # Check if files exist
    if not Path(CHUNKS_FILE).exists():
        print(f"\n❌ Chunks file not found: {CHUNKS_FILE}")
        print("\n💡 Update CHUNKS_FILE variable to point to your JSONL chunks file")
        print("   Example: CHUNKS_FILE = 'parsed_chunks.jsonl'\n")
        exit(1)
    
    if not Path(TEST_QUESTIONS).exists():
        print(f"\n❌ Test questions file not found: {TEST_QUESTIONS}")
        print("\n💡 Make sure test_questions.json is in the same directory\n")
        exit(1)
    
    # Create evaluator
    evaluator = RAGEvaluator(
        chunks_file=CHUNKS_FILE,
        faiss_index_path=FAISS_INDEX,
        test_questions_file=TEST_QUESTIONS
    )
    
    # Run evaluation
    print("\n🚀 Starting evaluation...")
    
    # Option: Evaluate subset first (faster)
    # results_df = evaluator.run_evaluation(max_questions=5)
    
    # Full evaluation
    results_df = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_results(results_df)
    
    print("\n✅ Evaluation complete!")
    print("\n💡 Next steps:")
    print("   1. Check the generated CSV for detailed metrics")
    print("   2. Review questions with low scores")
    print("   3. Improve retrieval/generation based on findings")
    print("   4. Re-run evaluation to track progress\n")
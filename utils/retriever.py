import requests
import numpy as np
from config import OLLAMA_URL, EMBEDDING_MODEL
from state import state
from utils.indexer import to_float16
import re

def perform_hybrid_search(query, k=60):
    with state.lock:
        if not state.vector_index or not state.bm25_index:
            return []
            
        # 1. Vector Search
        response = requests.post(f"{OLLAMA_URL}/api/embed",
                                     json={"model": EMBEDDING_MODEL, "input": query})
        
        embed_np = np.array(response.json()['embeddings'][0]).reshape(1, -1)
        D, I = state.vector_index.search(to_float16(embed_np), k)
        
        # 2. BM25 Search
        tokenized_query = re.findall(r'\w+', query.lower())
        bm25_scores = state.bm25_index.get_scores(tokenized_query)
        top_n_bm25 = np.argsort(bm25_scores)[::-1][:k]
        
        # 3. Fuse Rankings (RRF)
        final_scores = {}
        RRF_K = 60
        
        red_flags = ["political", "donation", "bribe", "gift", "trust", "conflict", "relative"]
        active_flags = [word for word in red_flags if word in query.lower()]

        def get_boost(chunk_idx):
            if not active_flags: return 0.0
            text = state.chunk_map.get(chunk_idx, "").lower()
            return 0.15 if any(flag in text for flag in active_flags) else 0.0

        for rank, idx in enumerate(I[0]):
            if idx == -1: continue
            if idx not in final_scores: final_scores[idx] = 0.0
            final_scores[idx] += (1.0 / (rank + RRF_K)) + get_boost(idx)
            
        for rank, idx in enumerate(top_n_bm25):
            if idx not in final_scores: final_scores[idx] = 0.0
            final_scores[idx] += (1.0 / (rank + RRF_K)) + get_boost(idx)
            
        # --- 4. NEW LOGIC: Dynamic Cutoff (The "Noise Gate") ---
        # Sort all candidates by score
        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_candidates:
            return []

        # Get the score of the absolute best match
        best_score = sorted_candidates[0][1]
        
        # Only keep chunks that are at least 50% as good as the winner
        # This deletes the "long tail" of garbage results
        filtered_results = []
        for idx, score in sorted_candidates:
            if score >= (best_score * 0.5):
                filtered_results.append((idx, score))
            
            # Stop if we have enough good ones (e.g., top 20 candidates max)
            if len(filtered_results) >= 20:
                break
                
        # Return the top k from the FILTERED list
        return filtered_results[:5]  # STRICTLY return 5
    
if __name__ == "__main__":
    test_query = "What are the ethics guidelines for accepting gifts?"
    results = perform_hybrid_search(test_query)
    print("Search Results (idx, score):", results)
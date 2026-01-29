import ollama
import faiss
import numpy as np
import os
from config import EMBEDDING_MODEL

# --- CONFIGURATION ---
# nomic-embed-text = 768
# mxbai-embed-large = 1024
# llama3.2 = 3072
EMBEDDING_DIM = 768 

def to_float16(arr):
    """Compress vectors to save RAM (Optional, keeps it fast)"""
    return arr.astype(np.float32)

def generate_embeddings_batch(texts, batch_size=10):
    """
    Generates embeddings for a list of texts using Ollama.
    Processes in batches to avoid crashing memory.
    """
    embeddings = []
    total = len(texts)
    
    print(f"   Using Embedding Model: {EMBEDDING_MODEL}")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            for text in batch:
                # IMPORTANT: Clean newlines to improve embedding quality
                clean_text = text.replace("\n", " ")
                response = ollama.embed(model=EMBEDDING_MODEL, input=clean_text)
                embeddings.append(response['embeddings'][0])
                
            print(f"   - Embedded {min(i + batch_size, total)}/{total} chunks...")
            
        except Exception as e:
            print(f" Error embedding batch {i}: {e}")
            # Insert zero-vectors for failed chunks to keep alignment
            for _ in batch:
                embeddings.append(np.zeros(EMBEDDING_DIM))

    return np.array(embeddings, dtype='float32')

def build_index_optimized(all_chunks, index_name="global_index"):
    """
    Takes the FULL list of chunks (History + New), embeds them,
    and builds a fresh FAISS index.
    """
    print(f"\n Indexer: Rebuilding memory for {len(all_chunks)} chunks...")
    
    if not all_chunks:
        print(" No chunks to index.")
        return None, {}

    # 1. Generate Embeddings for EVERYTHING
    # (We re-run this every time to ensure IDs map perfectly 1:1)
    vectors = generate_embeddings_batch(all_chunks)
    
    # 2. Verify Dimensions
    if vectors.shape[1] != EMBEDDING_DIM:
        print(f"Dimension Mismatch! Model gave {vectors.shape[1]}, expected {EMBEDDING_DIM}.")
    
    # 3. Create FAISS Index
    # IndexFlatL2 is Exact Search (Best for < 10k chunks)
    index = faiss.IndexFlatL2(vectors.shape[1]) 
    index.add(vectors)
    
    print(f"Index Built. Total Vectors: {index.ntotal}")

    # 4. Create Mapping (ID -> Text)
    # Since we embedded in order, ID 0 is Chunk 0, ID 1 is Chunk 1...
    chunk_map = {i: text for i, text in enumerate(all_chunks)}
    
    return index, chunk_map
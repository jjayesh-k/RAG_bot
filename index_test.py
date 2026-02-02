import faiss
import numpy as np
import ollama
import pickle
from rank_bm25 import BM25Okapi
from config import EMBEDDING_MODEL
import re

# --- CONFIG ---
JSONL_FILE = "q4fy22-presentation_phi35_clean.jsonl"  # <--- YOUR FILE
EMBEDDING_MODEL = "nomic-embed-text" # or "mxbai-embed-large" or "all-minilm"

import faiss
import numpy as np
import ollama
from config import EMBEDDING_MODEL

def to_float16(embed_np):
    return embed_np.astype(np.float32)

def simple_tokenize(text):
    # Uses regex to keep only alphanumeric words (removes '?', '.', '!')
    return re.findall(r'\w+', text.lower())

def build_rag_index(chunks):
    if not chunks:
        return None, None, {}

    print(f"📊 [Indexer] Processing {len(chunks)} chunks...")
    
    # --- 1. VECTOR INDEX (FAISS) ---
    all_embeddings = []
    BATCH_SIZE = 20
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend(response['embeddings'])
            print(f"   -> Embedded batch {i} to {i+len(batch)}")
        except Exception as e:
            print(f"❌ Error embedding batch {i}: {e}")

    if not all_embeddings:
        return None, None, {}

    dimension = len(all_embeddings[0])
    np_embeddings = np.array(all_embeddings).astype('float32')
    vector_index = faiss.IndexFlatL2(dimension)
    vector_index.add(np_embeddings)
    
    # --- 2. KEYWORD INDEX (BM25) ---
    print("🔤 [Indexer] Building BM25 Keyword Index...")
    tokenized_corpus = [simple_tokenize(doc) for doc in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Mapping
    chunk_map = {i: txt for i, txt in enumerate(chunks)}
    
    print("✅ [Indexer] Hybrid Index Ready.")
    return vector_index, bm25_index, chunk_map

if __name__ == "__main__":
    build_rag_index()
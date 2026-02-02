import json
import faiss
import numpy as np
import ollama
import os
import time

# --- CONFIG ---
JSONL_FILE = "q4fy22-presentation_phi35_clean.jsonl"  # <--- YOUR FILE
EMBEDDING_MODEL = "nomic-embed-text" # or "mxbai-embed-large" or "all-minilm"

import faiss
import numpy as np
import ollama
from config import EMBEDDING_MODEL

def to_float16(embed_np):
    return embed_np.astype(np.float32)

def build_rag_index(chunks):
    """
    Builds FAISS index IN MEMORY for the Flask App.
    Accepts a list of strings (chunks).
    Returns (index, mapping).
    """
    if not chunks:
        return None, {}

    print(f"📊 [Indexer] Embedding {len(chunks)} chunks...")
    
    all_embeddings = []
    
    # Batch Processing (Safe for 8GB RAM)
    BATCH_SIZE = 20 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            # Ollama handles lists of strings automatically
            response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
            batch_embeddings = response['embeddings']
            all_embeddings.extend(batch_embeddings)
            print(f"   -> Processed batch {i} to {i+len(batch)}")
        except Exception as e:
            print(f"❌ Error embedding batch {i}: {e}")

    if not all_embeddings:
        return None, {}

    # Convert to Numpy for FAISS
    dimension = len(all_embeddings[0])
    np_embeddings = np.array(all_embeddings).astype('float32')

    # Build Flat L2 Index
    index = faiss.IndexFlatL2(dimension)
    index.add(np_embeddings)
    
    # Create Mapping (Index ID -> Text)
    chunk_map = {i: txt for i, txt in enumerate(chunks)}
    
    print(f"✅ [Indexer] Index built with {index.ntotal} vectors.")
    return index, chunk_map

if __name__ == "__main__":
    build_rag_index()
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re
import requests
from config import EMBEDDING_MODEL

def to_float16(embed_np):
    """Helper to convert embeddings to float32 (FAISS requirement)"""
    return embed_np.astype(np.float32)

def simple_tokenize(text):
    """
    Enhanced Tokenizer:
    - Preserves numbers and acronyms
    - Handles compound words better
    """
    # First, extract the filename/source tag to avoid polluting tokens
    text = re.sub(r'\[.*?\]', '', text)  # Remove [Filename] tags
    
    # Split on whitespace and punctuation, but keep alphanumeric together
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return tokens

def build_rag_index(chunks):
    print(f"[Indexer] Using embedding model: {EMBEDDING_MODEL}")
    if not chunks:
        print("[Indexer] No chunks to index.")
        return None, None, {}

    print(f"[Indexer] Processing {len(chunks)} chunks...")
    
    # --- 1. SAFE EMBEDDING LOOP (Keeps Text & Vectors Synced) ---
    valid_embeddings = []
    valid_chunks = []  # We only keep text if it embeds successfully
    
    BATCH_SIZE = 10  # Reduced batch size to prevent timeouts
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_text = chunks[i : i + BATCH_SIZE]
        try:
            # Generate Embeddings
            # response = ollama.embed(model=EMBEDDING_MODEL, input=batch_text)
            # batch_vectors = response.get('embeddings', [])
            
            response = requests.post("http://tmlpnewskc31137.tmindia.tatamotors.com:11434/api/embed",
                                     json={"model": EMBEDDING_MODEL, "input": batch_text}) 
            
            # CRITICAL CHECK: Ensure we got exactly one vector per text chunk
            if response.status_code != 200:
                print(f"Error embedding batch {i}: {response.status_code}")
                continue
            x = response.json()
            batch_vectors = x.get('embeddings', [])
            
            # Ensure it is list of vectors
            if batch_vectors and isinstance(batch_vectors[0], float):
                batch_vectors = [batch_vectors]
            
            if len(batch_vectors) != len(batch_text):
                print(f"Mismatch in Batch {i}: Sent {len(batch_text)}, got {len(batch_vectors)}. Retrying one by one...")
                
                # FALLBACK: Try one by one to save valid chunks
                for single_chunk in batch_text:
                    try:
                        res = requests.post("http://tmlpnewskc31137.tmindia.tatamotors.com:11434/api/embed",
                                            json={"model": EMBEDDING_MODEL, "input": [single_chunk]})
                        if res.status_code == 200:
                            x = res.json()
                            vec = x.get('embeddings', [])
                            if vec:
                                valid_embeddings.extend(vec)
                                valid_chunks.append(single_chunk)
                    except:
                        pass # Skip only the bad chunk
                continue

            # If successful, add BOTH to the valid lists
            valid_embeddings.extend(batch_vectors)
            valid_chunks.extend(batch_text)
            print(f"   -> Successfully indexed batch {i} to {i+len(batch_text)}")
            
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            # If a batch fails, we skip BOTH text and vector. Alignment preserved.
            continue

    if not valid_embeddings:
        print("CRITICAL: No embeddings were generated. Check your Ollama model.")
        return None, None, {}

    # --- 2. BUILD FAISS INDEX ---
    dimension = len(valid_embeddings[0])
    np_embeddings = np.array(valid_embeddings).astype('float32')
    
    # vector_index = faiss.IndexFlatL2(dimension)
    vector_index = faiss.IndexHNSWFlat(dimension, 32)
    vector_index.add(np_embeddings)
    # print("----------->", vector_index)
    
    # --- 3. BUILD KEYWORD INDEX (BM25) ---
    # Use valid_chunks ONLY (to match FAISS IDs)
    print(f"🔤 [Indexer] Building BM25 Index for {len(valid_chunks)} valid docs...")
    tokenized_corpus = [simple_tokenize(doc) for doc in valid_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # --- 4. BUILD MAPPING ---
    # Map ID -> Text (1:1 relationship is now guaranteed)
    chunk_map = {i: txt for i, txt in enumerate(valid_chunks)}
    
    print(f"[Indexer] Index built successfully with {len(valid_chunks)} documents.")
    
    # Return the 3 objects app.py expects
    return vector_index, bm25_index, chunk_map

if __name__ == "__main__":
    test_chunks = [
        "EV battery degradation depends on temperature.",
        "Charging speed varies with SOC level."
    ]
    
    vector_index, bm25_index, chunk_map = build_rag_index(test_chunks)
        
    for i, text in chunk_map.items():
        tokens = simple_tokenize(text)
        print(f"\nChunk ID {i}")
        print("Text:", text)
        print("Tokens:", tokens)
        
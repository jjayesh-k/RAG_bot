from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import ollama
import time
import os
import numpy as np
import json
import threading
import re
from werkzeug.utils import secure_filename
from docling_parser import parse_hybrid_pdf
import tempfile
from recursive_chunker import chunk_markdown
from indexer import build_index_optimized, to_float16
from config import EMBEDDING_MODEL, LANGUAGE_MODEL
import sys
import webbrowser
from threading import Timer
import traceback # Added for detailed error logs

# --- CONFIGURATION ---
# Check if a settings.json file exists
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

settings_path = os.path.join(app_dir, 'settings.json')

if os.path.exists(settings_path):
    print(f"✓ Loading custom settings from {settings_path}")
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            EMBEDDING_MODEL = settings.get('EMBEDDING_MODEL', EMBEDDING_MODEL)
            LANGUAGE_MODEL = settings.get('LANGUAGE_MODEL', LANGUAGE_MODEL)
    except Exception as e:
        print(f"⚠️ Error loading settings: {e}")

print(f"Using Models -> Language: {LANGUAGE_MODEL} | Embedding: {EMBEDDING_MODEL}")
app = Flask(__name__)

# --- GLOBAL STATE ---
class RAGState:
    def __init__(self):
        self.vector_index = None
        self.chunk_map = {}
        self.all_chunks = [] 
        self.is_ready = False
        self.lock = threading.Lock()

state = RAGState()

# --- FLASK ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    start_time = time.time()
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    print("\n=== Processing Upload ===")
    TEMP_DIR = "temp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)
    new_chunks = []

    for f in files:
        filename = secure_filename(f.filename)
        print(f"Reading: {filename}")
        file_path = os.path.join(TEMP_DIR, filename)
        f.save(file_path)

        try:
            full_text = ""
            if filename.lower().endswith('.pdf'):
                full_text = parse_hybrid_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as txt_f:
                    full_text = txt_f.read()

            raw_chunks = chunk_markdown(full_text)
            clean_filename = filename.replace(".pdf", "").replace("_", " ")
            file_context = f"[{clean_filename}] "
            enhanced_chunks = [file_context + chunk for chunk in raw_chunks]
            new_chunks.extend(enhanced_chunks)
            print(f" - {filename}: {len(enhanced_chunks)} chunks created")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    with state.lock:
        state.all_chunks.extend(new_chunks)
        print(f"Building index with {len(state.all_chunks)} total chunks...")
        try:
            # Rebuild Index with ALL chunks
            index, mapping = build_index_optimized(state.all_chunks)
            state.vector_index = index
            state.chunk_map = mapping
            state.is_ready = True
        except Exception as e:
            print(f" CRITICAL INDEX ERROR: {e}")
            traceback.print_exc()

    return jsonify({
        "message": f"Added {len(new_chunks)} chunks. Total Memory: {len(state.all_chunks)} chunks.",
        "count": len(state.all_chunks)
    })

@app.route('/chat', methods=['POST'])
def chat():
    # DEBUG 1: Check if request hits the endpoint
    print("\n--- [DEBUG] 1. Chat Endpoint Reached ---")
    
    if not state.is_ready:
        print(" System not ready")
        return jsonify({"error": "System not ready. Upload files first."}), 400

    try:
        data = request.json
        query = data.get("message", "")
        print(f"--- Query: {query} ---")

        # DEBUG 2: Check Embedding
        print("--- [DEBUG] 2. Generatng Embedding... ---")
        with state.lock:
            try:
                response = ollama.embed(model=EMBEDDING_MODEL, input=query)
                # Check if embedding exists
                if not response.get('embeddings'):
                    raise ValueError("Ollama returned no embeddings!")
                
                embed_list = response['embeddings'][0]
                embed_np = np.array(embed_list).reshape(1, -1)
                
                print(f"--- [DEBUG] 3. Searching FAISS (Vector Dim: {embed_np.shape}) ---")
                
                # Check FAISS index status
                if state.vector_index is None:
                    raise ValueError("Vector Index is NONE!")
                
                D, I = state.vector_index.search(to_float16(embed_np), 30)
                print(f"--- [DEBUG] 4. Search Complete. Found {len(I[0])} matches ---")
                
            except Exception as e:
                print(f" VECTOR SEARCH FAILED: {e}")
                traceback.print_exc()
                return jsonify({"error": f"Internal Search Error: {str(e)}"}), 500

        # Logic: Hybrid Scoring
        results = []
        stop_words = {'what', 'is', 'explain', 'the', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'and', 'how', 'do', 'does', 'are'}
        query_terms = set([w for w in query.lower().split() if w not in stop_words])
        
        # Unpack FAISS results
        for idx, dist in zip(I[0], D[0]):
            if idx == -1: continue
            
            # Retrieve text
            chunk_text = state.chunk_map.get(idx, "")
            if not chunk_text: continue
            
            vector_score = 1 - dist
            keyword_bonus = 0
            
            # Simple keyword matching for scoring
            chunk_lower = chunk_text.lower()
            if any(q in chunk_lower for q in query_terms):
                keyword_bonus = 0.3
                
            final_score = vector_score + keyword_bonus
            results.append((idx, chunk_text, final_score))

        # Sort and Expand
        results.sort(key=lambda x: x[2], reverse=True)
        final_context_list = []
        seen_indices = set()
        
        # Top 3 + Neighbors
        for idx, txt, score in results[:3]:
            if idx in seen_indices: continue
            
            # Add Anchor
            final_context_list.append((txt, score))
            seen_indices.add(idx)
            
            # Add Neighbor (Next Chunk)
            next_idx = idx + 1
            if next_idx in state.chunk_map and next_idx not in seen_indices:
                 final_context_list.append((state.chunk_map[next_idx], score))
                 seen_indices.add(next_idx)

        print(f"--- [DEBUG] 5. Context Prepared: {len(final_context_list)} chunks ---")
        
        # Prepare Prompt
        context_str = "\n\n".join([f"{txt}" for txt, _ in final_context_list])
        
        def generate():
            try:
                # 1. Send Context Data to UI
                context_data = [{"text": txt[:200]+"...", "score": 1.0} for txt, _ in final_context_list[:5]]
                yield json.dumps({"type": "context", "data": context_data}) + "\n"
                
                # 2. Generate Answer
                system_instruction = "You are a helpful AI assistant. Answer the question based ONLY on the context provided."
                final_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                
                print("--- [DEBUG] 6. Starting Stream Generation... ---")
                stream = ollama.chat(
                    model=LANGUAGE_MODEL,
                    messages=[{'role': 'user', 'content': final_prompt}],
                    stream=True
                )
                
                for chunk in stream:
                    content = chunk['message']['content']
                    if content:
                        yield json.dumps({"type": "token", "content": content}) + "\n"
                        
                print("--- [DEBUG] 7. Stream Finished Successfully ---")
                
            except Exception as e:
                print(f" GENERATION ERROR: {e}")
                traceback.print_exc()
                # Try to send error to UI
                yield json.dumps({"type": "token", "content": f"\n[SYSTEM ERROR]: {str(e)}"}) + "\n"

        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

    except Exception as e:
        print(f" CRITICAL ROUTE ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_knowledge_base():
    with state.lock:
        state.vector_index = None
        state.chunk_map = {}
        state.all_chunks = []
        state.is_ready = False
    return jsonify({"message": "AI memory cleared!"})

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:8080/')
    
    print("Starting DEBUG RAG Engine...")
    Timer(1.5, open_browser).start()
    # DEBUG=True helps print errors, but use_reloader=False prevents double loading
    app.run(port=8080, debug=True, use_reloader=False)
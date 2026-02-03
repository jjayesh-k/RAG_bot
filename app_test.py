from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
import ollama
import time
import os
import numpy as np
import json
import threading
import re
from werkzeug.utils import secure_filename
import tempfile
import sys
import webbrowser
from threading import Timer
import traceback

# --- CUSTOM MODULES ---
# Using your existing filenames
from multi_parser_test import SmartMultiColumnParser 
from index_test import build_rag_index, to_float16
from config import EMBEDDING_MODEL, LANGUAGE_MODEL

# --- CONFIGURATION ---
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
CORS(app)

# --- GLOBAL STATE ---
class RAGState:
    def __init__(self):
        self.vector_index = None
        self.bm25_index = None   # <--- NEW: Stores the Keyword Index
        self.chunk_map = {}
        self.all_chunks = [] 
        self.is_ready = False
        self.lock = threading.Lock()

state = RAGState()

# --- HELPER: HYBRID SEARCH (RRF) ---
def perform_hybrid_search(query, k=60):
    with state.lock:
        if not state.vector_index or not state.bm25_index:
            return []
            
        # 1. Vector Search
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        embed_np = np.array(response['embeddings'][0]).reshape(1, -1)
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

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    start_time = time.time()
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    print("\n=== Processing Upload (Hybrid Search Enabled) ===")
    TEMP_DIR = "temp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Initialize Smart Parser
    parser = SmartMultiColumnParser(chunk_size=1000, chunk_overlap=200)
    new_chunks_text = []

    for f in files:
        filename = secure_filename(f.filename)
        print(f"📄 Reading: {filename}")
        file_path = os.path.join(TEMP_DIR, filename)
        f.save(file_path)

        try:
            # --- 1. PARSE & CHUNK ---
            # Check file type for PDF vs Text
            if filename.lower().endswith('.pdf'):
                raw_chunk_objects = parser.parse_and_chunk(file_path, verbose=True)
                clean_filename = filename.replace(".pdf", "").replace("_", " ")
                
                for chunk_obj in raw_chunk_objects:
                    # Tag format: [Filename] [Page X] Content...
                    formatted_text = (
                        f"[{clean_filename}] [Page {chunk_obj.page_num} | {chunk_obj.chunk_type}]\n"
                        f"{chunk_obj.content}"
                    )
                    new_chunks_text.append(formatted_text)
                print(f"   -> Extracted {len(raw_chunk_objects)} chunks from {filename}")
            else:
                # Text fallback
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as txt_f:
                    text = txt_f.read()
                    clean_filename = filename.replace("_", " ")
                    new_chunks_text.append(f"[{clean_filename}] {text}")
                print(f"   -> Added text file: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            traceback.print_exc()
        finally:
            time.sleep(0.1)
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except Exception as e: print(f"⚠️ Cleanup warning: {e}")

    # --- 2. INDEXING (Expects 3 Returns from Indexer) ---
    with state.lock:
        state.all_chunks.extend(new_chunks_text)
        print(f"📊 Building Hybrid Index with {len(state.all_chunks)} chunks...")
        
        try:
            # UNPACK 3 VALUES: Vector Index, BM25 Index, Mapping
            v_index, b_index, mapping = build_rag_index(state.all_chunks)
            
            state.vector_index = v_index
            state.bm25_index = b_index  # Store BM25 Index
            state.chunk_map = mapping
            state.is_ready = True
            
        except ValueError as ve:
            print("❌ Indexer Error: Did you update index_test.py to return 3 values?")
            print(f"Details: {ve}")
            return jsonify({"error": "Indexer mismatch. Please update index_test.py."}), 500
        except Exception as e:
            print(f"❌ CRITICAL INDEX ERROR: {e}")
            traceback.print_exc()

    elapsed_time = time.time() - start_time
    print(f"✅ Processing complete in {elapsed_time:.2f} seconds")

    return jsonify({
        "message": f"Successfully indexed {len(new_chunks_text)} new chunks.",
        "count": len(state.all_chunks),
        "processing_time": f"{elapsed_time:.2f}s"
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get("message", "").strip()

    # Greeting check
    if query.lower() in ['hi', 'hello', 'hey', 'greetings', 'hola']:
        def simple_stream():
            yield json.dumps({"type": "token", "content": "Hello! I am ready to answer questions about your uploaded documents."}) + "\n"
        return Response(stream_with_context(simple_stream()), mimetype='application/x-ndjson')

    if not state.is_ready:
        return jsonify({"error": "System not ready. Upload files first."}), 400

    try:
        # --- HYBRID SEARCH EXECUTION ---
        # We retrieve the top 15 chunks using the fused rankings
        hybrid_results = perform_hybrid_search(query, k=20)
        
        # Prepare Context
        top_k_chunks = []
        for idx, score in hybrid_results[:5]: # Take Top 15
            txt = state.chunk_map.get(idx, "")
            top_k_chunks.append((idx, txt, score))
            
        # Sort by ID to preserve reading order (helps LLM understand flow)
        top_k_chunks.sort(key=lambda x: x[0]) 
        
        context_str = "\n\n".join([txt for _, txt, _ in top_k_chunks])
        print(f"✅ Context Loaded: {len(top_k_chunks)} chunks for query: '{query}'")
        # --- DEBUG: PRINT RETRIEVED CHUNKS ---
        print("\n" + "="*50)
        print(f"🧐 LLM CONTEXT FOR QUERY: '{query}'")
        print("="*50)
        for i, (idx, txt, score) in enumerate(top_k_chunks):
            print(f"\n🔹 [Chunk {i+1} | Score: {score:.4f} | ID: {idx}]")
            print("-" * 30)
            # Print first 300 chars to keep it readable, remove newlines for clean log
            preview = txt[:3000000].replace('\n', ' ') + "..."
            print(preview)
        print("="*50 + "\n")
        # -------------------------------------
        
        
        # --- GENERATION ---
        def generate():
            # Send context preview to frontend (Optional debug)
            context_data = [{"text": txt[:200]+"...", "score": round(score, 4)} for _, txt, score in top_k_chunks[:5]]
            yield json.dumps({"type": "context", "data": context_data}) + "\n"

            sys_msg = (
                        "You are an Intelligent Knowledge Assistant. Your task is to answer the user's question using ONLY the provided Context chunks.\n\n"

                        "--- CRITICAL ANALYSIS RULES ---\n"
                        "1. DETECT THE INTENT:\n"
                        "   - IF asking 'CAN I...' or 'IS IT ALLOWED...': Check the rules. If prohibited, start with 'No' and explain why.\n"
                        "   - IF asking 'WHAT IS THE POLICY...' or 'STANCE': Do not just say 'No'. Summarize the full scope of the rule (e.g., who it applies to, specific prohibitions, and core principles).\n" 
                        "   - IF asking for DATA: Scan for exact keywords. The number immediately nearby is the answer.\n"
                        "   - IF asking for CONCEPTS: Synthesize a clear definition.\n"
    
                        "2. HANDLE PARSING ARTIFACTS:\n"
                        "   - Tables may look broken. If you see 'Value ... ... 10', the value is 10.\n"
                        "   - IMPORTANT: Always check sections marked '[ADDITIONAL NOTES / SIDEBARS]'. Vital headers and rules often appear there.\n"
    
                        "3. DATE & TERM MAPPING:\n"
                        "   - 'FY22' = '2022', 'Q1' = 'First Quarter'.\n"
    
                        "--- RESPONSE GUIDELINES ---\n"
                        "- BE COMPREHENSIVE FOR RULES: Mention key details like 'zero tolerance', 'agents', or 'intermediaries' if present in the text.\n"
                        "- BE PRECISE FOR DATA: Quote specific values.\n"
                        "- NO HALLUCINATIONS: If the answer is not in the text, strictly say 'I don't know'."
                    )
            user_msg = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
            
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': user_msg}],
                stream=True,
                options={"stop": ["Context:", "Question:", "User:", "System:", "\n\n\n"], "temperature": 0.1}
            )

            for chunk in stream:
                content = chunk['message']['content']
                if content:
                    yield json.dumps({"type": "token", "content": content}) + "\n"

        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_knowledge_base():
    with state.lock:
        state.vector_index = None
        state.bm25_index = None
        state.chunk_map = {}
        state.all_chunks = []
        state.is_ready = False
    return jsonify({"message": "AI memory cleared!"})

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')
        
    print("Starting Hybrid RAG Engine...")
    Timer(1.5, open_browser).start()
    app.run(port=5000, debug=True, use_reloader=False)
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import ollama
import time
import os
import numpy as np
import json
import threading
import re
from werkzeug.utils import secure_filename
from smart_parser import parse_hybrid_pdf
# from docling_parser import parse_hybrid_pdf
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
            # Small delay to ensure all file handles are released (Windows issue)
            time.sleep(0.1)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except PermissionError:
                # If still locked, try again after a longer delay
                time.sleep(0.5)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")

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

    elapsed_time = time.time() - start_time
    print(f"✅ Processing complete in {elapsed_time:.2f} seconds")

    return jsonify({
        "message": f"Added {len(new_chunks)} chunks. Total Memory: {len(state.all_chunks)} chunks.",
        "count": len(state.all_chunks),
        "processing_time": f"{elapsed_time:.2f}s"
    })

import re  # <--- ENSURE THIS IS IMPORTED AT THE TOP

import re  # Ensure this is imported at top of file

@app.route('/chat', methods=['POST'])
def chat():
    # 1. Parse Data
    data = request.json
    query = data.get("message", "").strip()

    # --- LEVEL 1: GREETINGS ---
    if query.lower() in ['hi', 'hello', 'hey', 'greetings', 'hola']:
        def simple_stream():
            yield json.dumps({"type": "token", "content": "Hello! I am ready to answer questions about your uploaded documents."}) + "\n"
        return Response(stream_with_context(simple_stream()), mimetype='application/x-ndjson')

    if not state.is_ready:
        return jsonify({"error": "System not ready. Upload files first."}), 400

    try:
        # 2. Embedding
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        embed_np = np.array(response['embeddings'][0]).reshape(1, -1)

        with state.lock:
            # Search deeper (Top 60) to find the right chunk across multiple files
            D, I = state.vector_index.search(to_float16(embed_np), 60)
            current_chunk_map = state.chunk_map.copy()

        # 3. SMART SCORING
        results = []
        common_tech_words = {'python', 'project', 'system', 'using', 'data', 'technologies', 'used', 'application', 'developed'}
        stop_words = {'what', 'is', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'and', 'how', 'do', 'does', 'are', 'tell', 'me'}
        
        words = re.findall(r'\w+', query.lower())
        query_terms = [w for w in words if w not in stop_words and len(w) > 2]

        for idx, dist in zip(I[0], D[0]):
            if idx == -1: continue
            chunk_text = current_chunk_map.get(idx, "")
            chunk_lower = chunk_text.lower()
            
            score = 1 - dist
            
            # Variable Boosting
            for term in query_terms:
                if term in chunk_lower:
                    if term in common_tech_words:
                        score += 0.5 
                    else:
                        score += 5.0 
            
            results.append((idx, chunk_text, float(score)))

        # Sort by Score
        results.sort(key=lambda x: x[2], reverse=True)

        # 4. CONTEXT WINDOW (Expanded for Multi-Doc)
        # CRITICAL FIX: Increased from 5 to 15 chunks.
        # This ensures that even if the Resume takes the top 5 spots,
        # the Policy chunks (at #6, #7...) will still be included.
        top_k_chunks = results[:15]
        top_k_chunks.sort(key=lambda x: x[0]) 

        context_str = "\n\n".join([txt for _, txt, _ in top_k_chunks])

        # 5. GENERATION
        def generate():
            context_data = [{"text": txt[:200]+"...", "score": score} for _, txt, score in top_k_chunks[:5]]
            yield json.dumps({"type": "context", "data": context_data}) + "\n"

            # --- SYSTEM PROMPT UPDATE ---
            # We explicitly teach the AI to look at the [Filename] tag.
            sys_msg = (
                "You are an intelligent document assistant. The Context below contains chunks from multiple different files. "
                "Each chunk starts with a tag like [Filename]. "
                "1. Identify which document is relevant to the user's question. "
                "2. If the user asks about 'Policy', ONLY use text from [Tata Code of Conduct]. Ignore [Resume]. "
                "3. If the user asks about 'Jayesh' or 'Projects', ONLY use text from [Resume]. Ignore [Code of Conduct]. "
                "4. If the answer is not in the correct document, say 'I don't know'."
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
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
# We use the new parser that handles Parsing AND Chunking together
from multi_parser_test import SmartMultiColumnParser 
from index_test import build_rag_index, build_index_optimized, to_float16
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
CORS(app) # Fixes CORS issues

# --- GLOBAL STATE ---
class RAGState:
    def __init__(self):
        self.vector_index = None
        self.chunk_map = {}
        self.all_chunks = [] 
        self.is_ready = False
        self.lock = threading.Lock()

state = RAGState()

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

    print("\n=== Processing Upload (Smart Method) ===")
    TEMP_DIR = "temp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Initialize the Smart Parser
    parser = SmartMultiColumnParser(chunk_size=1000, chunk_overlap=200)
    new_chunks_text = []

    for f in files:
        filename = secure_filename(f.filename)
        print(f"📄 Reading: {filename}")
        file_path = os.path.join(TEMP_DIR, filename)
        f.save(file_path)

        try:
            # --- 1. PARSE & CHUNK (The New Way) ---
            # This function now does BOTH: reads PDF and returns ready-made chunks
            # It returns a list of ParsedChunk objects
            raw_chunk_objects = parser.parse_and_chunk(file_path, verbose=True)
            
            # --- 2. FORMATTING ---
            # We need to turn objects into strings for the Vector DB
            # We add [Filename] tags here so the "Traffic Cop" prompt works
            clean_filename = filename.replace(".pdf", "").replace("_", " ")
            
            for chunk_obj in raw_chunk_objects:
                # Format: [Filename] [Page X] Content...
                formatted_text = (
                    f"[{clean_filename}] [Page {chunk_obj.page_num} | {chunk_obj.chunk_type}]\n"
                    f"{chunk_obj.content}"
                )
                new_chunks_text.append(formatted_text)

            print(f"   -> Extracted {len(raw_chunk_objects)} chunks from {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            traceback.print_exc()
        finally:
            # Cleanup temp file
            time.sleep(0.1)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

    # --- 3. INDEXING ---
    with state.lock:
        state.all_chunks.extend(new_chunks_text)
        print(f"📊 Building index with {len(state.all_chunks)} total chunks...")
        
        try:
            # Rebuild Index with ALL chunks using the Optimized Batch Indexer
            index, mapping = build_rag_index(state.all_chunks)
            state.vector_index = index
            state.chunk_map = mapping
            state.is_ready = True
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
        # 1. Embedding
        response = ollama.embed(model=EMBEDDING_MODEL, input=query)
        embed_np = np.array(response['embeddings'][0]).reshape(1, -1)

        with state.lock:
            # Deep Search (Top 60)
            D, I = state.vector_index.search(to_float16(embed_np), 60)
            current_chunk_map = state.chunk_map.copy()

        # 2. Smart Scoring & Filtering
        results = []
        # Boost keywords
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

        # 3. Context Window (Top 15 for Multi-Doc coverage)
        top_k_chunks = results[:15]
        top_k_chunks.sort(key=lambda x: x[0]) 

        context_str = "\n\n".join([txt for _, txt, _ in top_k_chunks])

        # 4. Generation with Traffic Cop Prompt
        def generate():
            context_data = [{"text": txt[:200]+"...", "score": score} for _, txt, score in top_k_chunks[:5]]
            yield json.dumps({"type": "context", "data": context_data}) + "\n"

            sys_msg = (
                "You are an intelligent document assistant. The Context below contains chunks from multiple different files. "
                "Each chunk starts with a tag like [Filename]. "
                "1. FIRST, identify which document matches the user's intent. "
                "2. If the question is about 'Policies', 'Rules', 'Gifts', or 'Code of Conduct', ONLY use text from [Tata Code of Conduct]. IGNORE [Jayesh Kandar] or [Resume]. "
                "3. If the question is about 'Jayesh', 'Internships', or 'Projects', ONLY use text from [Jayesh Kandar] or [Resume]. IGNORE [Code of Conduct]. "
                "4. Answer strictly based on the correct document. Do not mix information."
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
    
    print("Starting Optimized RAG Engine...")
    Timer(1.5, open_browser).start()
    app.run(port=8080, debug=True, use_reloader=False)
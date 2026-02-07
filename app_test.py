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
from flashrank import Ranker, RerankRequest

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
        self.is_processing = False
        self.lock = threading.Lock()
        self.chat_history = []
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        
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

def rewrite_query(user_question, history):
    """
    Uses the LLM to rewrite the user's question into a standalone search query
    based on the conversation history.
    """
    # If history is empty, no need to rewrite
    if not history:
        return user_question

    print("🔄 Rewriting query with history...")
    
    system_prompt = (
        "You are a Search Query Generator. Your task is to rephrase the User's last question "
        "into a keyword-optimized search query. You have access to the conversation history.\n\n"
        "--- CRITICAL RULES ---\n"
        "1. LOOK FOR TOPIC SHIFTS: If the user asks about a NEW concept (e.g., switching from 'Ethics' to 'Revenue'), "
        "   IGNORE the previous document context (dates, page numbers, document titles). Treat it as a fresh search.\n"
        "2. RESOLVE PRONOUNS ONLY: Only use history to define words like 'it', 'they', or 'the company'.\n"
        "3. NO HYPOTHETICALS: Do not add specific dates or page numbers from history unless the user explicitly mentioned them in the CURRENT question.\n"
        "4. KEEP IT BROAD: If the user asks 'What was Q4 revenue?', do NOT append a document name. Just output 'Q4 revenue'.\n"
        "5. OUTPUT: Output ONLY the search query text."
    )
    
    # Simple history string
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]]) # Keep last 4 turns
    
    prompt = f"History:\n{history_str}\n\nUser's Last Question: {user_question}\n\nRewritten Standalone Query:"
    
    response = ollama.chat(model=LANGUAGE_MODEL, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ])
    
    new_query = response['message']['content'].strip()
    print(f"✨ Original: '{user_question}' -> Rewritten: '{new_query}'")
    return new_query

# --- ROUTES ---
@app.route('/')
def home():
    # return render_template('index.html')
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # --- 1. SET BUSY FLAG ---
    with state.lock:
        state.is_processing = True

    try:
        start_time = time.time()
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files received"}), 400

        print("\n=== Processing Upload (Hybrid Search Enabled) ===")
        TEMP_DIR = "temp_uploads"
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Initialize Smart Parser
        parser = SmartMultiColumnParser(chunk_size=1000, chunk_overlap=400) # Ensure overlap is 400!
        new_chunks_text = []

        for f in files:
            filename = secure_filename(f.filename)
            print(f"📄 Reading: {filename}")
            file_path = os.path.join(TEMP_DIR, filename)
            f.save(file_path)

            try:
                # --- PARSE & CHUNK ---
                if filename.lower().endswith('.pdf'):
                    raw_chunk_objects = parser.parse_and_chunk(file_path, verbose=True)
                    clean_filename = filename.replace(".pdf", "").replace("_", " ")
                    
                    for chunk_obj in raw_chunk_objects:
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

        # --- 2. INDEXING ---
        with state.lock:
            state.all_chunks.extend(new_chunks_text)
            print(f"📊 Building Hybrid Index with {len(state.all_chunks)} chunks...")
            
            try:
                v_index, b_index, mapping = build_rag_index(state.all_chunks)
                
                state.vector_index = v_index
                state.bm25_index = b_index
                state.chunk_map = mapping
                state.is_ready = True
                
            except ValueError as ve:
                print(f"❌ Indexer Error: {ve}")
                return jsonify({"error": "Indexer mismatch."}), 500

        elapsed_time = time.time() - start_time
        print(f"✅ Processing complete in {elapsed_time:.2f} seconds")

        return jsonify({
            "message": f"Successfully indexed {len(new_chunks_text)} new chunks.",
            "count": len(state.all_chunks),
            "processing_time": f"{elapsed_time:.2f}s"
        })

    except Exception as e:
        print(f"❌ CRITICAL UPLOAD ERROR: {e}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # --- 3. TURN OFF BUSY FLAG (Always runs) ---
        with state.lock:
            state.is_processing = False
            print("🔓 System Released (Ready for Queries)")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    raw_query = data.get("message", "").strip()

    # Initialize history if it doesn't exist yet
    if not hasattr(state, 'chat_history'):
        state.chat_history = []

    # Greeting check
    if raw_query.lower() in ['hi', 'hello', 'hey', 'greetings', 'hola']:
        def simple_stream():
            yield json.dumps({"type": "token", "content": "Hello! I am ready to answer questions about your uploaded documents."}) + "\n"
        return Response(stream_with_context(simple_stream()), mimetype='application/x-ndjson')

    # --- 1. THE PATIENT WAIT LOOP (Queueing Logic) ---
    if state.is_processing:
        print(f"⏳ System is busy. Holding query: '{raw_query[:20]}...'")
        max_retries = 120  # Wait up to 60 seconds
        
        for _ in range(max_retries):
            time.sleep(0.5)
            if not state.is_processing:
                print("✅ Processing complete! Resuming query.")
                break
        else:
            return jsonify({"error": "System is overloaded or taking too long. Please try again."}), 503

    # --- 2. Standard Readiness Check ---
    if not state.is_ready:
        return jsonify({"error": "System not ready. Upload files first."}), 400

    try:
        # --- STEP A: REWRITE THE QUERY ---
        # We use the rewritten query for SEARCHING to fix "Context Pollution"
        search_query = rewrite_query(raw_query, state.chat_history)

        # --- STEP B: HYBRID SEARCH + RERANKING (The Quality Upgrade) ---
        print(f"🔎 SEARCHING FOR: {search_query}")
        
        # 1. Broad Search: Get Top 25 (We cast a wider net now)
        initial_results = perform_hybrid_search(search_query, k=25)
        
        # 2. Format for FlashRank
        passages = []
        for idx, score in initial_results:
            text_content = state.chunk_map.get(idx, "")
            if text_content:
                passages.append({
                    "id": idx,
                    "text": text_content,
                    "meta": {} 
                })

        # 3. Rerank! (The AI Grader)
        print(f"⚖️ Reranking {len(passages)} chunks...")
        rerank_request = RerankRequest(query=search_query, passages=passages)
        reranked_results = state.ranker.rerank(rerank_request)
        
        # 4. Select Top 5 High-Quality Survivors
        final_top_k = 5
        top_k_chunks = []
        
        for result in reranked_results[:final_top_k]:
            idx = result['id']
            txt = result['text']
            score = result['score']
            top_k_chunks.append((idx, txt, score))
        
        # Optional: Sort by ID to maintain document reading order in the context
        # top_k_chunks.sort(key=lambda x: x[0]) 

        print(f"✅ Context Loaded: {len(top_k_chunks)} chunks (Filtered from {len(initial_results)})")

        # --- DEBUG: PRINT RERANKED CHUNKS ---
        print("\n" + "="*50)
        print(f"🏆 TOP {len(top_k_chunks)} RERANKED CHUNKS")
        print("="*50)
        for i, (idx, txt, score) in enumerate(top_k_chunks):
            print(f"\n🔹 [Chunk #{idx}] (Relevance Score: {score:.4f})")
            print("-" * 30)
            print(txt.replace('\n', ' ')[:300] + "...") 
            print("-" * 30)
        print("="*50 + "\n")

        context_str = "\n\n".join([txt for _, txt, _ in top_k_chunks])

        # --- STEP C: GENERATION ---
        def generate():
            # Send context preview to frontend
            context_data = [{"text": txt[:200]+"...", "score": round(float(score), 4)} for _, txt, score in top_k_chunks]
            yield json.dumps({"type": "context", "data": context_data}) + "\n"

            sys_msg = (
                "You are an Intelligent Knowledge Assistant. Your task is to answer the user's question using ONLY the provided Context chunks.\n\n"
                "--- CRITICAL ANALYSIS RULES ---\n"
                "1. DETECT THE INTENT:\n"
                "   - IF asking 'CAN I...' or 'IS IT ALLOWED...': Check the rules. If prohibited, start with 'No' and explain why.\n"
                "   - IF asking 'WHAT IS THE POLICY...' or 'STANCE': Do not just say 'No'. Summarize the full scope of the rule.\n" 
                "   - IF asking for DATA: Scan for exact keywords. The number immediately nearby is the answer.\n"
                "   - IF asking for CONCEPTS: Synthesize a clear definition.\n"
                "2. HANDLE PARSING ARTIFACTS:\n"
                "   - Tables may look broken. If you see 'Value ... ... 10', the value is 10.\n"
                "   - IMPORTANT: Always check sections marked '[ADDITIONAL NOTES / SIDEBARS]'. Vital headers often appear there.\n"
                "3. DATE & TERM MAPPING:\n"
                "   - 'FY22' = '2022', 'Q1' = 'First Quarter'.\n"
                "--- RESPONSE GUIDELINES ---\n"
                "- BE COMPREHENSIVE FOR RULES: Mention key details like 'zero tolerance' if present.\n"
                "- BE PRECISE FOR DATA: Quote specific values.\n"
                "- NO HALLUCINATIONS: If the answer is not in the text, strictly say 'I don't know'."
            )
            
            # Use raw_query for natural tone, but context is now hyper-relevant
            user_msg = f"Context:\n{context_str}\n\nQuestion: {raw_query}\n\nAnswer:"
            
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': user_msg}],
                stream=True,
                options={"stop": ["Context:", "Question:", "User:", "System:", "\n\n\n"], "temperature": 0.1}
            )

            full_response_text = ""

            for chunk in stream:
                content = chunk['message']['content']
                if content:
                    full_response_text += content
                    yield json.dumps({"type": "token", "content": content}) + "\n"

            # --- STEP D: UPDATE HISTORY ---
            state.chat_history.append({"role": "user", "content": raw_query})
            state.chat_history.append({"role": "assistant", "content": full_response_text})
            
            if len(state.chat_history) > 10:
                state.chat_history = state.chat_history[-10:]

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
        webbrowser.open_new('http://127.0.0.1:8080/')
        
    print("Starting Hybrid RAG Engine...")
    Timer(1.5, open_browser).start()
    app.run(port=8080, debug=True, use_reloader=False)
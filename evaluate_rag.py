import requests
import json
import numpy as np

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8080/chat"
GROUND_TRUTH_FILE = "ground_truth.json"
K = 5  # We are checking the Top 5 results

def get_rag_context(question):
    """
    Sends a query to the running RAG app and extracts the retrieved chunks.
    """
    try:
        # Enable streaming to catch the first chunk (which contains context)
        response = requests.post(API_URL, json={"message": question}, stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    # Your app sends {"type": "context", "data": [...]} first
                    if data.get("type") == "context":
                        return data["data"] # Returns list of {"text": "...", "score": ...}
        return []
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return []

def evaluate_batch():
    # 1. Load Ground Truth
    try:
        with open(GROUND_TRUTH_FILE, 'r') as f:
            test_set = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {GROUND_TRUTH_FILE} not found. Please create it first.")
        return

    total_precision = []
    total_recall = []

    print(f"\n📊 STARTING EVALUATION (K={K})")
    print(f"   Testing {len(test_set)} questions...\n")
    print("-" * 60)

    # 2. Loop through questions
    for i, item in enumerate(test_set):
        question = item["question"]
        # ... inside the loop ...
        
        # FIX: Remove .pdf from expected filenames for matching
        expected_files = [f.replace(".pdf", "").replace("_", " ").lower() for f in item["expected_files"]]
        
        # Get actual results
        retrieved_chunks = get_rag_context(question)
        
        # DEBUG: See what we actually got!
        if i == 0: 
            print(f"\n[DEBUG] Q1 Retrieved: {[c['text'][:50] for c in retrieved_chunks[:3]]}")
        
        # If we got fewer than K chunks, we analyze what we have
        retrieved_k = retrieved_chunks[:K]
        
        # --- CALCULATE METRICS ---
        relevant_retrieved = 0
        
        # Check each retrieved chunk to see if it comes from the right file
        # (We assume the chunk text starts with [Filename] as per your app logic)
        for chunk in retrieved_k:
            chunk_text = chunk["text"].lower()
            
            # Check if ANY of the expected filenames are in this chunk
            is_relevant = any(f_name in chunk_text for f_name in expected_files)
            if is_relevant:
                relevant_retrieved += 1

        # A. Precision@K: (Relevant Found) / K
        # How much of the retrieved list was useful?
        # Note: If we retrieved fewer than K, we divide by len(retrieved_k) or K depending on strictness. 
        # Standard strict P@K divides by K.
        precision = relevant_retrieved / K if K > 0 else 0

        # B. Recall@K: (Relevant Found) / (Total Expected Files)
        # Did we find at least one chunk from every file we needed?
        # Note: This is a simplified "File Recall". 
        total_expected = len(expected_files)
        recall = relevant_retrieved / total_expected if total_expected > 0 else 0
        # Cap recall at 1.0 (in case we retrieved multiple chunks from the same file)
        recall = min(recall, 1.0)

        total_precision.append(precision)
        total_recall.append(recall)

        print(f"Q{i+1}: '{question}'")
        print(f"   Files Found: {relevant_retrieved}/{total_expected} needed")
        print(f"   👉 Precision@{K}: {precision:.2f} | Recall@{K}: {recall:.2f}")
        print("-" * 60)

    # 3. Final Report
    avg_p = np.mean(total_precision)
    avg_r = np.mean(total_recall)

    print("\n🏆 FINAL RESULTS")
    print("=" * 30)
    print(f"MEAN PRECISION@{K}: {avg_p:.1%}")
    print(f"MEAN RECALL@{K}:    {avg_r:.1%}")
    print("=" * 30)
    
    if avg_r < 0.7:
        print("⚠️  Advice: Your RECALL is low. Try increasing chunk overlap or using Hybrid Search.")
    elif avg_p < 0.5:
        print("⚠️  Advice: Your PRECISION is low. Your search is finding the answer but also a lot of junk.")
    else:
        print("✅ Excellent! Your system is robust.")

if __name__ == "__main__":
    evaluate_batch()
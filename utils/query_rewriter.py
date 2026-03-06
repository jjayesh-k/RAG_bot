from config import LANGUAGE_MODEL, OLLAMA_URL
import requests

def rewrite_query(user_question, history):
    """
    Uses the LLM to rewrite the user's question into a standalone search query
    based on the conversation history.
    """
    # If history is empty, no need to rewrite
    if not history:
        return user_question

    print("Rewriting query with history...")
    
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
    
    payload = {
    "model": LANGUAGE_MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    "stream": False
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120
    )

    data = response.json()
    # print("-------> data ", data)
    
    new_query = data["message"]["content"].strip()
    
    print(f"Original: '{user_question}' -> Rewritten: '{new_query}'")
    return new_query

if __name__ == "__main__":
    # Simple test
    history = [
        {"role": "user", "content": "What were the revenue numbers for Q4?"},
        {"role": "assistant", "content": "The revenue for Q4 was $10 million."},
        {"role": "user", "content": "How about the ethics section?"},
    ]
    rewrite_query("What about the ethics section?", history)
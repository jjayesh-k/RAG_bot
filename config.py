import os
import sys
import json

# DEFAULT SETTINGS (Fallbacks)
DEFAULT_CONFIG = {
    "embedding_model": "nomic-embed-text",
    "language_model": "phi3.5:3.8b-mini-instruct-q4_K_M",
    "batch_size": 50,
    "ollama_url": "http://tmlpnewskc31137.tmindia.tatamotors.com:11434"
}

# DETERMINE APPLICATION DIRECTORY
# Works for both Python script and compiled .exe
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# DEFINE PATHS
CACHE_DIR = os.path.join(APP_DIR, "cache")
INDEX_CACHE = os.path.join(APP_DIR, "index_cache")
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

# Ensure required folders exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDEX_CACHE, exist_ok=True)

# LOAD USER SETTINGS (settings.json)
CONFIG = DEFAULT_CONFIG.copy()

if os.path.exists(SETTINGS_PATH):
    try:
        with open(SETTINGS_PATH, "r") as f:
            user_settings = json.load(f)

        CONFIG.update(user_settings)

        print(f"Loaded custom config: {CONFIG['language_model']} / {CONFIG['embedding_model']}")

    except Exception as e:
        print(f"Error reading settings.json: {e}")

else:
    print("settings.json not found. Using defaults.")

# EXPORT CONSTANTS (Used across project)
EMBEDDING_MODEL = CONFIG["embedding_model"]
LANGUAGE_MODEL = CONFIG["language_model"]
BATCH_SIZE = CONFIG["batch_size"]
OLLAMA_URL = CONFIG["ollama_url"]

# DEBUG TEST
if __name__ == "__main__":
    print("APP_DIR:", APP_DIR)
    print("SETTINGS_PATH:", SETTINGS_PATH)
    print("Embedding Model:", EMBEDDING_MODEL)
    print("Language Model:", LANGUAGE_MODEL)
    print("Batch Size:", BATCH_SIZE)
    print("Ollama URL:", OLLAMA_URL)
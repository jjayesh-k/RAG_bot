import threading
from flashrank import Ranker

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
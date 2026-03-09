from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.engine import URL

# 1. Define the Base
Base = declarative_base()

# --- TABLES ---

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship: A user can have many chat sessions
    # cascade="all, delete" ensures if a user is deleted, their chats are too.
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(100), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False) # e.g., 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    session = relationship("ChatSession", back_populates="messages")

# --- DATABASE CONNECTION SETUP ---
# Replace with your actual PostgreSQL credentials
# Format: postgresql://username:password@localhost:5432/database_name
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="jayesh@456",
    host="localhost",
    port=5432,
    database="rag_app_db"
)

# Create the Engine and Session maker
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Creates all tables in the database if they don't exist yet."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database schema initialized successfully.")

if __name__ == "__main__":
    init_db()
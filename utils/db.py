from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import sessionmaker, scoped_session, DeclarativeBase, relationship
from utils.setting import settings
import datetime

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,               # 生产环境设为 False
    pool_pre_ping=True,       # 连接池预检
    pool_recycle=3600
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 用于 Streamlit 的线程安全会话（每个请求独立）
def get_db_session():
    return scoped_session(SessionLocal)

# ORM 基类
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False, default="新对话")
    agent_type = Column(String(50), nullable=False, default="base")  # 'base' | 'stock'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    chart_paths = Column(JSON, nullable=True)  # stock agent 的图表路径 {"kline": "...", ...}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

# 建表函数（供初始化脚本调用）
def create_tables():
    Base.metadata.create_all(bind=engine)
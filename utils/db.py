from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
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
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

# 建表函数（供初始化脚本调用）
def create_tables():
    Base.metadata.create_all(bind=engine)
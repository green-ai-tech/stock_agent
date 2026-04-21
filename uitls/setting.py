"""
    配置模块，
    主要配置：
        1. 模型与代理的参数
        2. RAG与向量数据库的参数
        3. 日志系统的阐述
    作者：LogicYe
    日期：2026/04/20
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from .paths import PROJECT_ROOT,ENV_FILE_PATH,VECTOR_STORE_PATH,LOG_PATH

class Settings(BaseSettings):
    """
    系统配置（通用路径版，任何设备、任何路径都能用）
    """
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,        
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    #====================1. 模型与代理的配置 =======================
    llm_model: str                = Field(default="ollama:gemma4:e4b", description="使用的LLM模型")
    llm_temperature: float        = Field(default=0.7, ge=0.0, le=2.0, description="温度")
    llm_max_tokens: Optional[int] = Field(default=512, description="最大token")
    llm_base_url: str             = Field(default="http://127.0.0.1/", description="模型地址")
    llm_streaming: bool           = Field(default=False, description="流式输出")

    #====================2. RAG与向量数据库的配置===================
    vector_store_dir: str         = Field(default=VECTOR_STORE_PATH,  description="向量数据库目录")
    embedding_model: str          = Field(default="qwen3-embedding:4b")
    chunk_size: int               = Field(default=1000)
    chunk_overlap: int            = Field(default=200)
    retriever_search_type: str    = Field(default="similarity")
    retriever_k: int              = Field(default=4)
    retriever_score_threshold: float = Field(default=0.5)
    retriever_fetch_k: int        = Field(default=20)


    #====================3.日志系统的配置==========================
    log_level: str                  = Field(default="INFO", description="日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL")
    log_file: str                   = Field(default=LOG_PATH, description="日志文件路径")
    log_rotation: str               = Field(default="100 MB", description="日志文件轮转大小")
    log_retention: str              = Field(default="30 days", description="日志文件保留时间")

#全局单例
settings = Settings()

if __name__ == "__main__":

    print("✅ 项目根目录：", PROJECT_ROOT)
    print("✅ .env 路径：", ENV_FILE_PATH)
    print("✅ 向量库路径：", settings.vector_store_dir)
    print("✅ LLM 模型：", settings.llm_model)  #观察路径
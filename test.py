import os
from dotenv import load_dotenv
import tushare as ts

load_dotenv()

ts_token = os.getenv('TUSHARE_TOKEN')
if ts_token:
    ts.set_token(ts_token)
    pro = ts.pro_api()
    df = pro.daily(ts_code='600519.SH', start_date='20240101', end_date='20240301')
    print(df.head())
else:
    print("TUSHARE_TOKEN not found in environment variables")

from langchain_ollama import ChatOllama
llm_base_url = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:11434')
llm_model = os.getenv('LLM_MODEL', 'qwen3.6:latest')
llm = ChatOllama(model=llm_model, base_url=llm_base_url)
print(llm.invoke("你好"))
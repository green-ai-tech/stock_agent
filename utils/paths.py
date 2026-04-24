"""
配置项目文件路径，专门解决相对路径与绝对路径的问题
"""
import os
from pathlib import Path

def find_project_root(marker_files=("requirements.txt", "pyproject.toml", ".env.example")):
    """通过向上查找标志文件来确定项目根目录"""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return str(parent)
    # 回退方案：如果标志文件都没找到，假设 paths.py 在 utils/ 下，返回父目录的父目录
    return str(current.parent.parent)

PROJECT_ROOT       = find_project_root()
ENV_FILE_PATH      = os.path.join(PROJECT_ROOT, ".env")
VECTOR_STORE_PATH  = os.path.join(PROJECT_ROOT, "data")
LOG_PATH           = os.path.join(PROJECT_ROOT, "logs/stock_agent.log")
STOCK_CHARTS_DIR   = os.path.join(PROJECT_ROOT, "imgs/stock")


def get_stock_charts_dir() -> Path:
    """获取股票图表保存目录（imgs/stock）"""
    charts_dir = Path(PROJECT_ROOT) / "imgs" / "stock"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


#=======================路径测试============================
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ENV_FILE_PATH:", ENV_FILE_PATH)
    print("VECTOR_STORE_PATH:", VECTOR_STORE_PATH)
    print("LOG_PATH:",LOG_PATH)







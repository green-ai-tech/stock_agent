# """
# 股票分析智能体工厂 - 复用模型工厂和提示词系统
# """
# from langchain.agents import create_agent
# from langchain.agents.structured_output import ToolStrategy

# from agents.models.base_models import get_model_by_preset
# from agents.prompts.system_prompt import get_system_prompt
# from agents.tools.stock_tools import get_stock_daily_data, plot_stock_charts, get_stock_basic_info
# from agents.models.stock_models import StockAnalysisOutput
# from utils.logger import logger

# def create_stock_analyst_agent():
#     """
#     创建股票分析智能体
#     使用 preset='structure' 模型（temperature=0.0）确保结构化输出稳定
#     """
#     logger.info("初始化股票分析智能体...")
    
#     # 复用现有模型工厂 - 选择结构化输出预设
#     model = get_model_by_preset(preset="structure")

#     # 获取股票分析师专用提示词（包含当前时间）
#     system_prompt = get_system_prompt(mode="stock_analyst", include_time=True)
    
#     tools = [
#         get_stock_daily_data,
#         plot_stock_charts,
#         get_stock_basic_info
#     ]
    
#     agent = create_agent(
#         model=model,
#         tools=tools,
#         system_prompt=system_prompt,
#         #response_format=ToolStrategy(schema=StockAnalysisOutput),
#     )
#     logger.info("股票分析智能体创建成功")
#     return agent

# agents/stock_agent.py
from langchain.agents import create_agent
from agents.models.base_models import get_model_by_preset
from agents.prompts.system_prompt import get_system_prompt
from agents.tools.stock_tools import get_stock_daily_data, plot_stock_charts, get_stock_basic_info
from utils.logger import logger

def create_stock_analyst_agent():
    logger.info("初始化股票分析智能体...")
    # 使用 creative 预设，temperature 较高，鼓励模型生成内容
    model = get_model_by_preset(preset="creative")  # 或者 "default"
    system_prompt = get_system_prompt(mode="stock_analyst", include_time=True)
    tools = [get_stock_daily_data, plot_stock_charts, get_stock_basic_info]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        # 不设置 response_format
    )
    logger.info("股票分析智能体创建成功")
    return agent
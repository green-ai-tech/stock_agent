"""
股票分析智能体工厂 - 复用模型工厂和提示词系统
"""
from langchain.agents import create_agent
from agents.models.base_models import get_model_by_preset
from agents.prompts.system_prompt import get_system_prompt
from agents.tools.stock_tools import get_stock_daily_data, plot_stock_charts, get_stock_basic_info
from utils.logger import logger

def create_stock_analyst_agent():
    """
    创建股票分析智能体
    使用 structure 预设模型（temperature=0.0）确保输出稳定
    """
    logger.info("初始化股票分析智能体...")
    model = get_model_by_preset(preset="structure")
    system_prompt = get_system_prompt(mode="stock_analyst", include_time=True)
    tools = [get_stock_daily_data, plot_stock_charts, get_stock_basic_info]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    logger.info("股票分析智能体创建成功")
    return agent

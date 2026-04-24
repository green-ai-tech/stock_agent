"""
股票分析结构化输出模型
"""
from pydantic import BaseModel, Field
from typing import Dict

class StockAnalysisOutput(BaseModel):
    """股票分析结构化输出模型"""
    stock_code: str         = Field(description="股票代码 (如: 600519.SH)")
    stock_name: str         = Field(description="股票名称")
    analysis_date: str      = Field(description="分析日期，格式 YYYY-MM-DD HH:MM:SS，北京时间")
    
    current_price: float    = Field(description="最新收盘价")
    price_change_pct: float = Field(description="涨跌幅(%)")
    volume: float           = Field(description="成交量(手)")
    amount: float           = Field(description="成交额(万元)")
    
    ma5: float              = Field(description="5日均线")
    ma20: float             = Field(description="20日均线")
    ma60: float             = Field(description="60日均线")
    volatility: float       = Field(description="波动率(20日标准差%)")
    
    max_price: float        = Field(description="分析周期内最高价")
    min_price: float        = Field(description="分析周期内最低价")
    avg_volume: float       = Field(description="平均成交量(手)")
    total_return: float     = Field(description="周期总收益率(%)")
    
    technical_comment: str  = Field(description="技术面评论")
    volume_comment: str     = Field(description="量能评论")
    risk_comment: str       = Field(description="风险评论")
    
    recommendation: str     = Field(description="投资建议")
    confidence_level: str   = Field(description="信心等级: 高/中/低")
    key_support: float      = Field(description="关键支撑位")
    key_resistance: float   = Field(description="关键阻力位")
    
    charts: Dict[str, str]  = Field(default_factory=dict, description="图表路径字典")
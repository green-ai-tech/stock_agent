"""
股票分析智能体 - 基于LangChain 1.2 create_agent
功能：获取股票数据、生成K线图/趋势图/饼图、提供专业分析评论和投资建议
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field
import tushare as ts

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
import matplotlib

matplotlib.use('Agg')
# ============================================
# 1. 配置部分
# ============================================

TS_TOKEN = "cf8c570c7475d05a5ed12474997859b1ee4e17f85a4cc2ec3050610a"
ts.set_token(TS_TOKEN)
pro = ts.pro_api()


# 中文字体设置 (解决Matplotlib中文显示问题)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================
# 2. 定义结构化输出模型 (Pydantic)
# ============================================

class StockAnalysisOutput(BaseModel):
    """股票分析结构化输出模型"""
    
    # 基础信息
    stock_code: str         = Field(description="股票代码 (如: 600519.SH)")
    stock_name: str         = Field(description="股票名称")
    analysis_date: str      = Field(description="分析日期")
    
    # 核心指标
    current_price: float    = Field(description="最新收盘价")
    price_change_pct: float = Field(description="涨跌幅(%)")
    volume: float           = Field(description="成交量(手)")
    amount: float           = Field(description="成交额(万元)")
    
    # 技术指标
    ma5: float          = Field(description="5日均线")
    ma20: float         = Field(description="20日均线")
    ma60: float         = Field(description="60日均线")
    volatility: float   = Field(description="波动率(20日标准差%)")
    
    # 统计指标
    max_price: float    = Field(description="分析周期内最高价")
    min_price: float    = Field(description="分析周期内最低价")
    avg_volume: float   = Field(description="平均成交量(手)")
    total_return: float = Field(description="周期总收益率(%)")
    
    # 分析评论 (大模型生成)
    technical_comment: str  = Field(description="技术面评论: 分析K线形态、均线排列、趋势判断")
    volume_comment: str     = Field(description="量能评论: 分析成交量变化、量价配合情况")
    risk_comment: str       = Field(description="风险评论: 识别潜在风险点和警示信号")
    
    # 投资建议
    recommendation: str     = Field(description="投资建议: 买入/持有/减持/卖出，并说明理由")
    confidence_level: str   = Field(description="信心等级: 高/中/低，基于数据完整性和信号一致性")
    key_support: float      = Field(description="关键支撑位")
    key_resistance: float   = Field(description="关键阻力位")
    
    # 图表路径
    charts: Dict[str, str]  = Field(default_factory=dict,description="生成的图表文件路径: kline, trend, pie")


# ============================================
# 3. 工具函数定义
# ============================================

@tool
def get_stock_daily_data(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"],
    days: Annotated[int, "获取数据的天数，默认120"] = 120
) -> str:
    """
    获取股票日线数据。
    返回包含日期、开盘、收盘、最高、最低、成交量、成交额等信息的DataFrame字符串表示。
    """
    try:
        end_date    = datetime.now().strftime('%Y%m%d')
        start_date  = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,open,high,low,close,vol,amount'
        )
        
        if df is None or df.empty:
            return f"未获取到股票 {ts_code} 的数据，请检查股票代码是否正确"
        
        # 数据预处理
        df['trade_date']    = pd.to_datetime(df['trade_date'])
        df                  = df.sort_values('trade_date')
        df['vol']           = df['vol'] / 100  # 转换为手
        df['amount']        = df['amount'] / 10000  # 转换为万元
        
        # 计算技术指标
        df['ma5']           = df['close'].rolling(window=5).mean()
        df['ma20']          = df['close'].rolling(window=20).mean()
        df['ma60']          = df['close'].rolling(window=60).mean()
        df['returns']       = df['close'].pct_change() * 100
        df['volatility']    = df['returns'].rolling(window=20).std()
        
        # 保存到全局变量供绘图使用
        global _stock_data_cache
        _stock_data_cache = df
        
        # 返回最新数据的摘要
        latest = df.iloc[-1]
        summary = f"""    # ToolMessgae 或者字符串
股票代码: {ts_code}
数据周期: {start_date} 至 {end_date} (共{len(df)}个交易日)

【最新交易日数据】{latest['trade_date'].strftime('%Y-%m-%d')}:
- 收盘价: {latest['close']:.2f}
- 开盘价: {latest['open']:.2f}
- 最高价: {latest['high']:.2f}
- 最低价: {latest['low']:.2f}
- 成交量: {latest['vol']:.0f}手
- 成交额: {latest['amount']:.2f}万元

【技术指标】:
- 5日均线: {latest['ma5']:.2f}
- 20日均线: {latest['ma20']:.2f}
- 60日均线: {latest['ma60']:.2f}
- 20日波动率: {latest['volatility']:.2f}% (如有数据)

【近期表现】:
- 近5日涨跌幅: {((df['close'].iloc[-1]  / df['close'].iloc[-6] - 1) * 100) if len(df) >= 6 else 0:.2f}%
- 近20日涨跌幅: {((df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100) if len(df) >= 21 else 0:.2f}%
- 区间最高价: {df['high'].max():.2f}
- 区间最低价: {df['low'].min():.2f}
"""
        return summary
        
    except Exception as e:
        return f"获取股票数据时出错: {str(e)}"


@tool
def plot_stock_charts(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"],
    stock_name: Annotated[Optional[str], "股票名称，用于图表标题"] = None
) -> str:
    """
    绘制股票分析图表，包括K线图、趋势图和成交量饼图。
    返回生成的图表文件路径。
    """
    print(">>>>>")
    try:
        global _stock_data_cache
        
        if _stock_data_cache is None or _stock_data_cache.empty:
            return "请先调用 get_stock_daily_data 获取数据"
        
        df = _stock_data_cache.copy()
        stock_name = stock_name or ts_code
        
        # 创建图表保存目录
        os.makedirs("stock_charts", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        chart_paths = {}
        
        # ---------- 图1: K线图 (使用matplotlib简化版) ----------
        fig1, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
        
        # K线子图
        ax1 = axes[0]
        dates = df['trade_date'].values
        
        # 绘制K线 (简化: 用细线表示价格区间，用粗线表示开盘收盘)
        for i, (idx, row) in enumerate(df.iterrows()):
            # 实体颜色: 收盘>=开盘为红色(涨)，否则绿色(跌)
            color = 'red' if row['close'] >= row['open'] else 'green'
            # 最高-最低线 (影线)
            ax1.plot([dates[i], dates[i]], [row['low'], row['high']], 
                    color=color, linewidth=0.8)
            # 开盘-收盘实体
            ax1.plot([dates[i], dates[i]], [row['open'], row['close']], 
                    color=color, linewidth=4)
        
        # 绘制均线
        if 'ma5' in df.columns:
            ax1.plot(dates, df['ma5'], 'b-', linewidth=1.2, label='MA5', alpha=0.8)
        if 'ma20' in df.columns:
            ax1.plot(dates, df['ma20'], 'orange', linewidth=1.2, label='MA20', alpha=0.8)
        if 'ma60' in df.columns:
            ax1.plot(dates, df['ma60'], 'purple', linewidth=1.2, label='MA60', alpha=0.8)
        
        ax1.set_title(f'{stock_name} ({ts_code}) - K线图', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格 (元)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 成交量子图
        ax2 = axes[1]
        colors = ['red' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'green' 
                  for i in range(len(df))]
        ax2.bar(dates, df['vol'], color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('成交量 (手)')
        ax2.set_xlabel('交易日期')
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        kline_path = f"stock_charts/kline_{ts_code}_{timestamp}.png"
        plt.savefig(kline_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['kline'] = kline_path
        
        # ---------- 图2: 趋势图 (价格+成交量+均线) ----------
        fig2, ax = plt.subplots(figsize=(14, 8))
        
        # 双Y轴
        ax2_twin = ax.twinx()
        
        # 价格线
        ax.plot(dates, df['close'], 'r-', linewidth=2, label='收盘价', marker='o', markersize=3)
        
        # 均线
        if 'ma5' in df.columns:
            ax.plot(dates, df['ma5'], 'b--', linewidth=1.5, label='MA5', alpha=0.7)
        if 'ma20' in df.columns:
            ax.plot(dates, df['ma20'], 'orange', linewidth=1.5, label='MA20', alpha=0.7)
        
        # 成交量柱状图(右轴)
        ax2_twin.bar(dates, df['vol'], color='gray', alpha=0.3, width=0.8, label='成交量')
        ax2_twin.set_ylabel('成交量 (手)', color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        
        ax.set_title(f'{stock_name} ({ts_code}) - 价格趋势与成交量', fontsize=14, fontweight='bold')
        ax.set_xlabel('交易日期')
        ax.set_ylabel('价格 (元)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        trend_path = f"stock_charts/trend_{ts_code}_{timestamp}.png"
        plt.savefig(trend_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['trend'] = trend_path
        
        # ---------- 图3: 成交量分布饼图 (按价格区间) ----------
        fig3, axes_pie = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图: 按成交量大小分类
        vol_desc = df['vol'].describe()
        vol_categories = ['极高 (>75%分位)', '较高 (50-75%分位)', 
                         '中等 (25-50%分位)', '较低 (<25%分位)']
        vol_counts = [
            len(df[df['vol'] > vol_desc['75%']]),
            len(df[(df['vol'] > vol_desc['50%']) & (df['vol'] <= vol_desc['75%'])]),
            len(df[(df['vol'] > vol_desc['25%']) & (df['vol'] <= vol_desc['50%'])]),
            len(df[df['vol'] <= vol_desc['25%']])
        ]
        
        axes_pie[0].pie(vol_counts, labels=vol_categories, autopct='%1.1f%%', 
                       colors=['#ff6b6b', '#ffa502', '#26de81', '#45aaf2'])
        axes_pie[0].set_title('成交量分布', fontsize=12, fontweight='bold')
        
        # 右图: 涨跌分布
        returns = df['returns'].dropna()
        up_days = len(returns[returns > 0])
        down_days = len(returns[returns < 0])
        flat_days = len(returns[returns == 0])
        
        axes_pie[1].pie([up_days, down_days, flat_days], 
                       labels=[f'上涨 ({up_days}天)', f'下跌 ({down_days}天)', f'平盘 ({flat_days}天)'],
                       autopct='%1.1f%%', colors=['#26de81', '#ff6b6b', '#dfe6e9'])
        axes_pie[1].set_title('涨跌分布', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'{stock_name} ({ts_code}) - 数据分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pie_path = f"stock_charts/pie_{ts_code}_{timestamp}.png"
        plt.savefig(pie_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths['pie'] = pie_path
        
        # 保存图表路径到全局变量
        global _chart_paths_cache
        _chart_paths_cache = chart_paths
        
        return f"图表已生成:\n- K线图: {kline_path}\n- 趋势图: {trend_path}\n- 饼图: {pie_path}"
        
    except Exception as e:
        return f"绘制图表时出错: {str(e)}"


@tool
def get_stock_basic_info(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"]
) -> str:
    """
    获取股票基础信息，包括股票名称、行业、上市日期等。
    """
    try:
        # 获取股票基本信息
        df_basic = pro.stock_basic(
            ts_code=ts_code,
            fields='ts_code,name,industry,list_date,market'
        )
        
        if df_basic is None or df_basic.empty:
            return f"未找到股票 {ts_code} 的基础信息"
        
        stock = df_basic.iloc[0]
        
        # 获取公司概况
        try:
            df_company = pro.daily_basic(ts_code=ts_code, 
                                         trade_date=datetime.now().strftime('%Y%m%d'),
                                         fields='pe,pe_ttm,pb,total_mv,circ_mv')
            if df_company is not None and not df_company.empty:
                pe = df_company.iloc[0]['pe']
                pb = df_company.iloc[0]['pb']
                total_mv = df_company.iloc[0]['total_mv'] / 10000  # 转换为亿元
            else:
                pe, pb, total_mv = 'N/A', 'N/A', 'N/A'
        except:
            pe, pb, total_mv = 'N/A', 'N/A', 'N/A'
        
        info = f"""
【股票基础信息】
- 股票代码: {stock['ts_code']}
- 股票名称: {stock['name']}
- 所属行业: {stock['industry']}
- 上市日期: {stock['list_date']}
- 交易市场: {stock['market']}

【估值指标】
- 市盈率PE: {pe if pe != 'N/A' else '暂无数据'}
- 市净率PB: {pb if pb != 'N/A' else '暂无数据'}
- 总市值: {total_mv if total_mv != 'N/A' else '暂无数据'} 亿元
"""
        return info
        
    except Exception as e:
        return f"获取基础信息时出错: {str(e)}"


# 全局缓存变量
_stock_data_cache = None
_chart_paths_cache = None


# ============================================
# 4. 创建智能体
# ============================================

# 专业系统提示词
SYSTEM_PROMPT = """你是一位专业的股票分析师，拥有超过15年的A股市场投资经验。你的任务是：
1. 基于沪深股市近120个工作日的交易数据，对股票进行全面分析
2. 提供客观、专业的投资建议
3. 识别潜在风险和投资机会

分析框架：
- 技术面：
    - 关注均线系统(5/20/60日线)的排列关系
    - K线形态
    - 趋势方向
- 量能面：
    - 成交量变化趋势
    - 量价配合关系
    - 异常放量/缩量信号
- 风险识别：
    - 高位放量滞涨
    - 均线死叉
    - 跌破关键支撑等警示信号

工作流程：
- 获取最近交易日的数据
- 获取基础信息
- 生成分析图表
- 基于以上数据，给出专业的投资建议和技术分析。

注意事项：
- 如果数据不足(少于20个交易日)，请在评论中明确说明
- 建议应基于具体数据，避免主观臆测
- 风险提示是必要的，投资建议应包含止损位参考
- 分析逻辑要严密，前后逻辑一致


可用的工具有：
1. 使用 get_stock_daily_data 获取最近交易日的数据
2. 使用 plot_stock_charts 生成分析图表
3. 使用 get_stock_basic_info 获取基础信息


请使用工具获取数据，然后生成结构化的分析报告。请确保输出符合结构化格式要求。"""


def create_stock_analyst_agent():
    """创建股票分析智能体"""
    
    # 工具列表
    tools = [
        get_stock_daily_data,
        plot_stock_charts,
        get_stock_basic_info
    ]
    
    # 创建智能体，配置结构化输出
    model = init_chat_model(
        model="ollama:qwen3.6:latest",
        temperature=0.0,       # 对于结构化输出，建议temperature=0
        max_tokens=2048,        # 成本考虑
        base_url="http://192.168.8.21:11434/"
    )
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=StockAnalysisOutput),
    )
    
    return agent


# ============================================
# 5. 运行示例
# ============================================

def analyze_stock(ts_code: str):
    """
    分析单只股票
    
    Args:
        ts_code: Tushare股票代码，如 '600519.SH'
        stock_name: 股票名称，可选
    """
    print(f"开始分析股票: {ts_code}")
    print(f"{'='*60}\n")
    
    agent = create_stock_analyst_agent()
    
    # 构建分析请求
    user_message = f"""
请对股票 {ts_code} 进行完整的技术分析。
"""
    
    try:
        # 调用智能体
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
        # 提取结构化输出
        if "structured_response" in result:
            analysis: StockAnalysisOutput = result["structured_response"]
            
            # 打印分析报告
            print("\n" + "="*60)
            print(f"【股票分析报告】{analysis.stock_name} ({analysis.stock_code})")
            print(f"分析日期: {analysis.analysis_date}")
            print("="*60)
            
            print(f"\n📊 核心数据")
            print(f"  最新价格: {analysis.current_price:.2f} 元")
            print(f"  涨跌幅: {analysis.price_change_pct:.2f}%")
            print(f"  成交量: {analysis.volume:.0f} 手")
            print(f"  成交额: {analysis.amount:.2f} 万元")
            print(f"  周期涨幅: {analysis.total_return:.2f}%")
            
            print(f"\n📈 技术指标")
            print(f"  MA5: {analysis.ma5:.2f} | MA20: {analysis.ma20:.2f} | MA60: {analysis.ma60:.2f}")
            print(f"  波动率: {analysis.volatility:.2f}%")
            print(f"  支撑位: {analysis.key_support:.2f} | 阻力位: {analysis.key_resistance:.2f}")
            
            print(f"\n💬 技术面评论")
            print(f"  {analysis.technical_comment}")
            
            print(f"\n📊 量能评论")
            print(f"  {analysis.volume_comment}")
            
            print(f"\n⚠️ 风险提示")
            print(f"  {analysis.risk_comment}")
            
            print(f"\n🎯 投资建议")
            print(f"  建议: {analysis.recommendation}")
            print(f"  信心等级: {analysis.confidence_level}")
            
            print(f"\n📁 生成图表")
            for chart_type, path in analysis.charts.items():
                print(f"  {chart_type}: {path}")
            
            print("\n" + "="*60)
            print("⚠️ 免责声明: 本分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
            
            return analysis
        else:
            # 回退：打印原始响应
            print("未能获取结构化输出，显示原始响应:")
            print(result)
            return None
            
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# 6. 主程序入口
# ============================================

if __name__ == "__main__":
    # 示例: 分析贵州茅台
    # 其他示例: '000858.SZ' (五粮液), '000001.SZ' (平安银行), '300750.SZ' (宁德时代)
    
    result = analyze_stock(ts_code="600519.SH")
    print("*" * 80, "\n", result)
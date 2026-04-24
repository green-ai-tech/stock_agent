"""
股票分析工具函数集
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Annotated, Optional
from langchain_core.tools import tool
import tushare as ts

from utils.logger import get_logger
logger = get_logger(__name__)
from utils.setting import settings
from utils.paths import get_stock_charts_dir
from utils.plot_helper import setup_matplotlib_style

# 初始化 Tushare
ts.set_token(settings.tushare_token)
pro = ts.pro_api()

# 设置 matplotlib 样式
setup_matplotlib_style()


@tool
def get_stock_daily_data(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"],
    days: Annotated[int, "获取数据的天数，默认120"] = 120
) -> str:
    """获取股票日线数据，返回包含技术指标的数据摘要"""
    try:
        logger.info(f"[Tool:get_stock_daily_data] 📊 开始获取数据: ts_code={ts_code}, days={days}")
        t0 = datetime.now()
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,open,high,low,close,vol,amount'
        )
        
        if df is None or df.empty:
            msg = f"未获取到股票 {ts_code} 的数据，请检查股票代码是否正确"
            logger.warning(msg)
            return msg
        
        # 数据预处理
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        df['vol'] = df['vol'] / 100  # 转换为手
        df['amount'] = df['amount'] / 10000  # 转换为万元
        
        # 计算技术指标
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        df['returns'] = df['close'].pct_change() * 100
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 将数据暂存到临时文件中，供绘图工具使用
        cache_dir = get_stock_charts_dir() / ".data_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{ts_code}_latest.parquet"
        df.to_parquet(cache_file)
        logger.debug(f"数据已缓存至: {cache_file}")
        
        # 返回最新数据的摘要
        latest = df.iloc[-1]
        summary = f"""股票代码: {ts_code}
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
- 近5日涨跌幅: {((df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100) if len(df) >= 6 else 0:.2f}%
- 近20日涨跌幅: {((df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100) if len(df) >= 21 else 0:.2f}%
- 区间最高价: {df['high'].max():.2f}
- 区间最低价: {df['low'].min():.2f}
"""
        logger.info(f"股票数据获取成功: {ts_code}")
        return summary
        
    except Exception as e:
        logger.error(f"获取股票数据时出错: {str(e)}", exc_info=True)
        return f"获取股票数据时出错: {str(e)}"


@tool
def plot_stock_charts(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"],
    stock_name: Annotated[Optional[str], "股票名称，用于图表标题"] = None
) -> str:
    """
    绘制股票分析图表，包括K线图、趋势图和成交量饼图。
    返回生成的图表文件路径（绝对路径）。
    """
    try:
        logger.info(f"开始生成图表: {ts_code}")
        # 尝试从缓存读取数据
        cache_dir = get_stock_charts_dir() / ".data_cache"
        cache_file = cache_dir / f"{ts_code}_latest.parquet"
        
        # 如果缓存不存在，则主动获取数据（避免依赖 get_stock_daily_data 先执行）
        if not cache_file.exists():
            logger.info(f"缓存不存在，主动获取数据: {ts_code}")
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,open,high,low,close,vol,amount'
            )
            if df is None or df.empty:
                return f"未获取到股票 {ts_code} 的数据，请检查股票代码是否正确"
            # 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df['vol'] = df['vol'] / 100
            df['amount'] = df['amount'] / 10000
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            df['returns'] = df['close'].pct_change() * 100
            df['volatility'] = df['returns'].rolling(window=20).std()
            # 保存缓存
            cache_dir.mkdir(exist_ok=True)
            df.to_parquet(cache_file)
            logger.info(f"数据已缓存至: {cache_file}")
        else:
            df = pd.read_parquet(cache_file)
        
        stock_name = stock_name or ts_code
        
        # 创建图表保存目录
        charts_dir = get_stock_charts_dir()
        charts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ---------- 图1: K线图 ----------
        fig1, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax1 = axes[0]
        dates = df['trade_date'].values
        
        for i, (idx, row) in enumerate(df.iterrows()):
            color = 'red' if row['close'] >= row['open'] else 'green'
            ax1.plot([dates[i], dates[i]], [row['low'], row['high']], color=color, linewidth=0.8)
            ax1.plot([dates[i], dates[i]], [row['open'], row['close']], color=color, linewidth=4)
        
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
        
        ax2 = axes[1]
        colors = ['red' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'green' for i in range(len(df))]
        ax2.bar(dates, df['vol'], color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('成交量 (手)')
        ax2.set_xlabel('交易日期')
        ax2.grid(True, alpha=0.3)
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        kline_path = charts_dir / f"kline_{ts_code}_{timestamp}.png"
        plt.savefig(kline_path, dpi=settings.stock_charts_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"K线图已保存: {kline_path}")
        
        # ---------- 图2: 趋势图 ----------
        fig2, ax = plt.subplots(figsize=(14, 8))
        ax2_twin = ax.twinx()
        ax.plot(dates, df['close'], 'r-', linewidth=2, label='收盘价', marker='o', markersize=3)
        if 'ma5' in df.columns:
            ax.plot(dates, df['ma5'], 'b--', linewidth=1.5, label='MA5', alpha=0.7)
        if 'ma20' in df.columns:
            ax.plot(dates, df['ma20'], 'orange', linewidth=1.5, label='MA20', alpha=0.7)
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
        trend_path = charts_dir / f"trend_{ts_code}_{timestamp}.png"
        plt.savefig(trend_path, dpi=settings.stock_charts_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"趋势图已保存: {trend_path}")
        
        # ---------- 图3: 饼图 ----------
        fig3, axes_pie = plt.subplots(1, 2, figsize=(14, 6))
        vol_desc = df['vol'].describe()
        vol_categories = ['极高 (>75%分位)', '较高 (50-75%分位)', '中等 (25-50%分位)', '较低 (<25%分位)']
        vol_counts = [
            len(df[df['vol'] > vol_desc['75%']]),
            len(df[(df['vol'] > vol_desc['50%']) & (df['vol'] <= vol_desc['75%'])]),
            len(df[(df['vol'] > vol_desc['25%']) & (df['vol'] <= vol_desc['50%'])]),
            len(df[df['vol'] <= vol_desc['25%']])
        ]
        axes_pie[0].pie(vol_counts, labels=vol_categories, autopct='%1.1f%%', colors=['#ff6b6b', '#ffa502', '#26de81', '#45aaf2'])
        axes_pie[0].set_title('成交量分布', fontsize=12, fontweight='bold')
        
        returns = df['returns'].dropna()
        up_days = len(returns[returns > 0])
        down_days = len(returns[returns < 0])
        flat_days = len(returns[returns == 0])
        axes_pie[1].pie([up_days, down_days, flat_days], labels=[f'上涨 ({up_days}天)', f'下跌 ({down_days}天)', f'平盘 ({flat_days}天)'], autopct='%1.1f%%', colors=['#26de81', '#ff6b6b', '#dfe6e9'])
        axes_pie[1].set_title('涨跌分布', fontsize=12, fontweight='bold')
        plt.suptitle(f'{stock_name} ({ts_code}) - 数据分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pie_path = charts_dir / f"pie_{ts_code}_{timestamp}.png"
        plt.savefig(pie_path, dpi=settings.stock_charts_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"饼图已保存: {pie_path}")
        
        # 返回绝对路径
        return f"图表已生成:\n- K线图: {str(kline_path)}\n- 趋势图: {str(trend_path)}\n- 饼图: {str(pie_path)}"
        
    except Exception as e:
        logger.error(f"绘制图表时出错: {str(e)}", exc_info=True)
        return f"绘制图表时出错: {str(e)}"


@tool
def get_stock_basic_info(
    ts_code: Annotated[str, "股票代码，如 '600519.SH'"]
) -> str:
    """
    获取股票基础信息，包括股票名称、行业、上市日期、估值指标等。
    """
    try:
        logger.info(f"获取股票基础信息: {ts_code}")
        df_basic = pro.stock_basic(
            ts_code=ts_code,
            fields='ts_code,name,industry,list_date,market'
        )
        if df_basic is None or df_basic.empty:
            return f"未找到股票 {ts_code} 的基础信息"
        
        stock = df_basic.iloc[0]
        
        # 获取估值指标
        try:
            today = datetime.now().strftime('%Y%m%d')
            df_company = pro.daily_basic(ts_code=ts_code, trade_date=today,
                                         fields='pe,pe_ttm,pb,total_mv,circ_mv')
            if df_company is not None and not df_company.empty:
                pe = df_company.iloc[0]['pe']
                pb = df_company.iloc[0]['pb']
                total_mv = df_company.iloc[0]['total_mv'] / 10000  # 亿元
            else:
                pe, pb, total_mv = 'N/A', 'N/A', 'N/A'
        except Exception as e:
            logger.warning(f"获取估值数据失败: {e}")
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
        logger.info(f"基础信息获取成功: {ts_code}")
        return info
        
    except Exception as e:
        logger.error(f"获取基础信息时出错: {str(e)}", exc_info=True)
        return f"获取基础信息时出错: {str(e)}"
"""
股市数据仪表盘
显示主要指数、K 线图、市场概况等信息
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from utils.logger import logger
from utils.setting import settings
from utils.plot_helper import setup_matplotlib_style

setup_matplotlib_style()

# 尝试导入 Tushare（延迟初始化）
try:
    import tushare as ts
    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()
    TUSHARE_AVAILABLE = True
except Exception:
    TUSHARE_AVAILABLE = False
    logger.warning("Tushare 未配置，将使用模拟数据")


# ── 指数代码映射 ──────────────────────────────────────────────
INDEX_MAP = {
    "上证指数": {"ts_code": "000001.SH", "market": "A股"},
    "深证成指": {"ts_code": "399001.SZ", "market": "A股"},
    "创业板指": {"ts_code": "399006.SZ", "market": "A股"},
    "恒生指数": {"ts_code": None, "market": "港股"},
    "标普500": {"ts_code": None, "market": "美股"},
}


# ── 数据获取函数 ─────────────────────────────────────────────
def _fetch_index_daily(ts_code: str) -> dict | None:
    """通过 Tushare 获取单个指数最新交易日数据"""
    if not TUSHARE_AVAILABLE:
        return None
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        df = pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="trade_date,open,close,high,low,vol,amount,pct_chg",
        )
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date")
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        change = latest["close"] - prev["close"] if prev is not None else 0
        change_pct = latest["pct_chg"] if prev is not None else 0
        return {
            "close": latest["close"],
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "change": change,
            "change_pct": change_pct,
            "vol": latest["vol"],
            "amount": latest["amount"],
            "trade_date": latest["trade_date"],
        }
    except Exception as e:
        logger.warning(f"获取指数 {ts_code} 失败: {e}")
        return None


def _generate_mock_index(name: str) -> dict:
    """为无 API 的指数生成模拟数据"""
    base_values = {
        "恒生指数": 19800.0,
        "标普500": 5200.0,
    }
    base = base_values.get(name, 3000.0)
    np.random.seed(hash(name + datetime.now().strftime("%Y%m%d")) % 2**31)
    change_pct = np.random.uniform(-2.0, 2.0)
    close = base * (1 + change_pct / 100)
    return {
        "close": round(close, 2),
        "open": round(base * (1 + np.random.uniform(-0.5, 0.5) / 100), 2),
        "high": round(close * (1 + abs(np.random.uniform(0, 0.8)) / 100), 2),
        "low": round(close * (1 - abs(np.random.uniform(0, 0.8)) / 100), 2),
        "change": round(close - base, 2),
        "change_pct": round(change_pct, 2),
        "vol": np.random.randint(1000000, 5000000),
        "amount": np.random.randint(800000, 2000000) * 1e8,
        "trade_date": datetime.now().strftime("%Y%m%d"),
        "is_mock": True,
    }


def _fetch_market_overview() -> dict:
    """获取市场概况（涨跌家数、成交额）"""
    if TUSHARE_AVAILABLE:
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
            # 尝试获取 A 股当日涨跌统计
            df = pro.daily(
                start_date=start_date,
                end_date=end_date,
                fields="trade_date,pct_chg",
            )
            if df is not None and not df.empty:
                latest_date = df["trade_date"].max()
                latest = df[df["trade_date"] == latest_date]
                up_count = int((latest["pct_chg"] > 0).sum())
                down_count = int((latest["pct_chg"] < 0).sum())
                flat_count = int((latest["pct_chg"] == 0).sum())
                # 获取总成交额
                df_amount = pro.daily(
                    start_date=latest_date,
                    end_date=latest_date,
                    fields="amount",
                )
                total_amount = df_amount["amount"].sum() if df_amount is not None else 0
                return {
                    "up": up_count,
                    "down": down_count,
                    "flat": flat_count,
                    "total_amount": total_amount / 1e8,  # 转换为亿元
                    "trade_date": latest_date,
                    "is_real": True,
                }
        except Exception as e:
            logger.warning(f"获取市场概况失败: {e}")

    # 模拟数据
    np.random.seed(int(datetime.now().strftime("%Y%m%d")))
    total = 5300
    up = np.random.randint(1500, 3500)
    down = np.random.randint(1500, 3500)
    flat = total - up - down
    return {
        "up": up,
        "down": down,
        "flat": flat,
        "total_amount": round(np.random.uniform(8000, 15000), 2),
        "trade_date": datetime.now().strftime("%Y%m%d"),
        "is_real": False,
    }


def _load_all_indices() -> list[dict]:
    """加载所有指数数据"""
    results = []
    for name, info in INDEX_MAP.items():
        ts_code = info["ts_code"]
        if ts_code and TUSHARE_AVAILABLE:
            data = _fetch_index_daily(ts_code)
            if data:
                data["name"] = name
                data["market"] = info["market"]
                results.append(data)
                continue
        # 回退到模拟数据
        data = _generate_mock_index(name)
        data["name"] = name
        data["market"] = info["market"]
        results.append(data)
    return results


# ── K 线图数据 ───────────────────────────────────────────────
def _fetch_kline_data(ts_code: str, days: int = 60) -> pd.DataFrame | None:
    """获取指数 K 线历史数据"""
    if TUSHARE_AVAILABLE:
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days + 15)).strftime("%Y%m%d")
            df = pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="trade_date,open,close,high,low,vol",
            )
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").tail(days)
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.reset_index(drop=True)
                return df
        except Exception as e:
            logger.warning(f"获取 K 线数据失败: {e}")

    # 模拟数据
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    np.random.seed(42)
    close = 3200 + np.cumsum(np.random.randn(days) * 15)
    df = pd.DataFrame({
        "trade_date": dates,
        "close": close,
        "open": close + np.random.randn(days) * 8,
        "high": close + abs(np.random.randn(days) * 12),
        "low": close - abs(np.random.randn(days) * 12),
        "vol": np.random.randint(200000, 600000, days),
    })
    return df


def _render_kline_chart(df: pd.DataFrame, title: str):
    """渲染 K 线图"""
    if df is None or df.empty:
        st.warning("暂无 K 线数据")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#9ca3af")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2d2d3f")
        ax.spines["left"].set_color("#2d2d3f")

    ax_price = axes[0]
    dates = df["trade_date"].values

    for i, row in df.iterrows():
        color = "#ef4444" if row["close"] >= row["open"] else "#22c55e"
        ax_price.plot([dates[i], dates[i]], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax_price.plot([dates[i], dates[i]], [row["open"], row["close"]], color=color, linewidth=4)

    # 均线
    if len(df) >= 5:
        ax_price.plot(dates, df["close"].rolling(5).mean(), color="#3b82f6", linewidth=1.2, label="MA5", alpha=0.8)
    if len(df) >= 20:
        ax_price.plot(dates, df["close"].rolling(20).mean(), color="#f59e0b", linewidth=1.2, label="MA20", alpha=0.8)

    ax_price.set_title(title, fontsize=14, fontweight="bold", color="#e5e7eb", pad=10)
    ax_price.set_ylabel("点位", color="#9ca3af")
    ax_price.legend(loc="upper left", facecolor="#1e1e2e", edgecolor="#2d2d3f", labelcolor="#e5e7eb")
    ax_price.grid(True, alpha=0.15, color="#9ca3af")
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha="right", color="#9ca3af")

    # 成交量
    ax_vol = axes[1]
    colors = ["#ef4444" if df.iloc[i]["close"] >= df.iloc[i]["open"] else "#22c55e" for i in range(len(df))]
    ax_vol.bar(dates, df["vol"], color=colors, alpha=0.6, width=0.8)
    ax_vol.set_ylabel("成交量", color="#9ca3af")
    ax_vol.set_xlabel("日期", color="#9ca3af")
    ax_vol.grid(True, alpha=0.15, color="#9ca3af")
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax_vol.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=45, ha="right", color="#9ca3af")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_pie_chart(overview: dict):
    """渲染涨跌分布饼图"""
    up = overview["up"]
    down = overview["down"]
    flat = overview["flat"]

    if up + down + flat == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    sizes = [up, down, flat]
    labels = [f"上涨\n{up}家", f"下跌\n{down}家", f"平盘\n{flat}家"]
    colors = ["#ef4444", "#22c55e", "#6b7280"]
    explode = (0.05, 0.05, 0.02)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "#e5e7eb", "fontsize": 11},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white")
        t.set_fontweight("bold")

    ax.set_title("涨跌分布", fontsize=14, fontweight="bold", color="#e5e7eb", pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── 渲染函数 ─────────────────────────────────────────────────
def _render_index_card(data: dict):
    """渲染单个指数卡片"""
    name = data["name"]
    close = data["close"]
    change_pct = data["change_pct"]
    change = data["change"]
    is_mock = data.get("is_mock", False)

    # 颜色
    if change_pct > 0:
        color = "#ef4444"  # 红涨
        arrow = "▲"
    elif change_pct < 0:
        color = "#22c55e"  # 绿跌
        arrow = "▼"
    else:
        color = "#6b7280"
        arrow = "—"

    mock_tag = " <span style='font-size:0.7em;color:#9ca3af;'>[模拟]</span>" if is_mock else ""

    st.markdown(
        f"""
        <div style="
            background: #1e1e2e;
            border-radius: 12px;
            padding: 1.2rem 1rem;
            text-align: center;
            border: 1px solid #2d2d3f;
            height: 100%;
        ">
            <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:0.3rem;">
                {name}{mock_tag}
            </div>
            <div style="font-size:1.6rem;font-weight:700;color:{color};margin-bottom:0.4rem;">
                {close:,.2f}
            </div>
            <div style="font-size:1rem;color:{color};font-weight:600;">
                {arrow} {change_pct:+.2f}%
            </div>
            <div style="font-size:0.8rem;color:#9ca3af;margin-top:0.2rem;">
                {change:+,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_market_stats(overview: dict):
    """渲染市场概况统计"""
    up = overview["up"]
    down = overview["down"]
    flat = overview["flat"]
    total = up + down + flat
    total_amount = overview["total_amount"]
    is_real = overview.get("is_real", False)

    up_pct = up / total * 100 if total > 0 else 0
    down_pct = down / total * 100 if total > 0 else 0
    flat_pct = flat / total * 100 if total > 0 else 0

    data_tag = "实时" if is_real else "模拟"

    st.markdown(
        f"""
        <div style="
            background: #1e1e2e;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #2d2d3f;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <span style="font-size:1rem;font-weight:600;color:#e5e7eb;">市场概况</span>
                <span style="font-size:0.75rem;color:#9ca3af;">{data_tag}</span>
            </div>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:120px;">
                    <div style="font-size:0.8rem;color:#9ca3af;">上涨家数</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#ef4444;">{up}</div>
                    <div style="font-size:0.75rem;color:#9ca3af;">{up_pct:.1f}%</div>
                </div>
                <div style="flex:1;min-width:120px;">
                    <div style="font-size:0.8rem;color:#9ca3af;">下跌家数</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#22c55e;">{down}</div>
                    <div style="font-size:0.75rem;color:#9ca3af;">{down_pct:.1f}%</div>
                </div>
                <div style="flex:1;min-width:120px;">
                    <div style="font-size:0.8rem;color:#9ca3af;">平盘家数</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#6b7280;">{flat}</div>
                    <div style="font-size:0.75rem;color:#9ca3af;">{flat_pct:.1f}%</div>
                </div>
                <div style="flex:1;min-width:120px;">
                    <div style="font-size:0.8rem;color:#9ca3af;">总成交额</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#3b82f6;">{total_amount:,.0f}</div>
                    <div style="font-size:0.75rem;color:#9ca3af;">亿元</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ── 主入口 ───────────────────────────────────────────────────
def ui_stock_analysis():
    """股市数据仪表盘"""

    # 标题
    st.title("股市数据分析", width="stretch", text_alignment="center")
    st.caption(
        f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        text_alignment="center",
    )

    # 刷新按钮
    col_btn_left, col_btn_center, col_btn_right = st.columns([4, 2, 4])
    with col_btn_center:
        if st.button("🔄 刷新数据", width="stretch"):
            st.rerun()

    st.divider()

    # ── 主要指数 ──
    st.markdown("### 📈 主要指数")

    with st.spinner("正在加载指数数据..."):
        indices = _load_all_indices()

    # 两行布局：第一行 3 个 A 股指数，第二行 2 个境外指数
    a_share = [d for d in indices if d["market"] == "A股"]
    foreign = [d for d in indices if d["market"] != "A股"]

    if a_share:
        cols = st.columns(len(a_share))
        for i, data in enumerate(a_share):
            with cols[i]:
                _render_index_card(data)

    if foreign:
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        cols = st.columns(len(foreign))
        for i, data in enumerate(foreign):
            with cols[i]:
                _render_index_card(data)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── 上证指数 K 线图 ──
    st.markdown("### 📉 上证指数 K 线图")

    with st.spinner("正在加载 K 线数据..."):
        kline_df = _fetch_kline_data("000001.SH", days=60)

    if kline_df is not None:
        _render_kline_chart(kline_df, "上证指数 (000001.SH) — 近 60 交易日")
    else:
        st.warning("暂无 K 线数据")

    st.divider()

    # ── 市场概况 ──
    st.markdown("### 📊 市场概况")

    with st.spinner("正在获取市场数据..."):
        overview = _fetch_market_overview()

    col1, col2 = st.columns([3, 2])
    with col1:
        _render_market_stats(overview)
    with col2:
        _render_pie_chart(overview)

    st.divider()

    # ── 底部说明 ──
    st.caption(
        "数据来源: Tushare Pro（A股实时）| 港股/美股为模拟数据，仅供参考。",
        text_alignment="center",
    )

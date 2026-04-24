"""
知识库管理页面
- 上传文档（PDF/TXT）
- 查看已入库文档列表
- 删除文档
- 预置知识文档导入
"""
import streamlit as st
from pathlib import Path
from rag.document_loader import load_uploaded_file, load_document
from rag.text_splitter import split_text
from rag.vector_store import add_documents, list_documents, delete_document, clear_all
from utils.logger import logger


# 预置知识文档内容
PRESET_KNOWLEDGE = {
    "技术分析基础.txt": """# 技术分析基础

## K线形态

### 锤子线（Hammer）
出现在下跌趋势末端，下影线较长（实体2倍以上），上影线很短或没有。
含义：空方力量衰竭，多方开始反击，是潜在的反转信号。

### 上吊线（Hanging Man）
出现在上涨趋势末端，形态与锤子线相同，但位置不同。
含义：多方力量减弱，空方开始发力，是潜在的见顶信号。

### 十字星（Doji）
开盘价等于或接近收盘价，上下影线较长。
含义：多空力量均衡，市场犹豫不决，可能是趋势反转的信号。

### 乌云盖顶（Dark Cloud Cover）
第一根为阳线，第二根为阴线，阴线开盘价高于前一根最高价，收盘价深入前一根实体一半以上。
含义：上涨趋势可能结束，空方开始占优。

### 早晨之星（Morning Star）
第一根为大阴线，第二根为小实体（可阴可阳），第三根为大阳线且收盘价深入第一根实体。
含义：下跌趋势反转信号。

### 红三兵（Three White Soldiers）
连续三根阳线，每根收盘价创新高。
含义：强烈的上涨信号，多方占绝对优势。

## 均线系统

### MA（移动平均线）
- MA5：5日均线，反映短期趋势
- MA20：20日均线，反映中期趋势
- MA60：60日均线，反映长期趋势

### 金叉与死叉
- 金叉：短期均线上穿长期均线，买入信号
- 死叉：短期均线下穿长期均线，卖出信号

### 多头排列与空头排列
- 多头排列：MA5 > MA20 > MA60，强势行情
- 空头排列：MA5 < MA20 < MA60，弱势行情

## 常用技术指标

### MACD（指数平滑异同移动平均线）
- DIF = 快线EMA(12) - 慢线EMA(26)
- DEA = DIF的9日EMA
- MACD柱 = 2 * (DIF - DEA)

使用方法：
1. DIF上穿DEA为金叉，买入信号
2. DIF下穿DEA为死叉，卖出信号
3. MACD柱由负转正，多方增强
4. 底背离：价格创新低但MACD不创新低，潜在反弹

### KDJ（随机指标）
- K值 = (收盘价 - N日最低价) / (N日最高价 - N日最低价) * 100
- D值 = K值的3日移动平均
- J值 = 3K - 2D

使用方法：
1. K > 80为超买区，< 20为超卖区
2. K上穿D为金叉，买入信号
3. J > 100为超买，< 0为超卖

### RSI（相对强弱指标）
- RSI = N日内上涨总幅度 / (上涨总幅度 + 下跌总幅度) * 100

使用方法：
1. RSI > 70为超买，< 30为超卖
2. 底背离：价格创新低但RSI不创新低
3. 顶背离：价格创新高但RSI不创新高

### 布林带（BOLL）
- 中轨 = 20日均线
- 上轨 = 中轨 + 2倍标准差
- 下轨 = 中轨 - 2倍标准差

使用方法：
1. 价格触及上轨，可能回调
2. 价格触及下轨，可能反弹
3. 带宽收窄后放大，趋势可能加速
""",
    "基本面分析.txt": """# 基本面分析

## 财务指标解读

### 盈利能力指标
- 毛利率 = (营业收入 - 营业成本) / 营业收入 * 100%
  - 反映产品竞争力，越高越好
  - 行业对比更有意义

- 净利率 = 净利润 / 营业收入 * 100%
  - 反映整体盈利能力
  - 通常 > 10% 为较好

- ROE（净资产收益率）= 净利润 / 净资产 * 100%
  - 反映股东投资回报率
  - 巴菲特认为连续5年 > 15% 为优秀

### 估值指标
- PE（市盈率）= 股价 / 每股收益
  - 越低说明越便宜，但要结合行业
  - 成长股PE通常较高

- PB（市净率）= 股价 / 每股净资产
  - PB < 1 可能被低估
  - 银行等重资产行业常用

- PEG = PE / 净利润增长率
  - PEG < 1 可能被低估
  - 适合成长股估值

### 偿债能力指标
- 资产负债率 = 总负债 / 总资产 * 100%
  - 一般 < 60% 为安全
  - 金融行业例外

- 流动比率 = 流动资产 / 流动负债
  - > 2 为安全
  - < 1 有短期偿债风险

## 行业分析框架

### 波特五力模型
1. 现有竞争者的竞争程度
2. 潜在新进入者的威胁
3. 替代品的威胁
4. 供应商的议价能力
5. 购买者的议价能力

### 行业生命周期
- 初创期：高增长、高风险、竞争少
- 成长期：快速增长、竞争加剧
- 成熟期：增长放缓、竞争激烈、龙头优势明显
- 衰退期：需求下降、产能过剩

## 公司分析要点

### 护城河类型
1. 品牌优势（如茅台、苹果）
2. 成本优势（规模效应、技术专利）
3. 网络效应（微信、支付宝）
4. 转换成本（企业软件、银行账户）
5. 政策壁垒（牌照、资源独占）

### 管理层评估
1. 诚信度：是否有不良记录
2. 能力：过往业绩、行业经验
3. 激励：股权激励、利益绑定
4. 资本配置：再投资 vs 分红 vs 回购
""",
    "风险管理.txt": """# 投资风险管理

## 仓位管理

### 固定比例法
每只股票不超过总资金的一定比例（如10%-20%）。
优点：简单易行，分散风险。

### 凯利公式
最优仓位 = (胜率 * 赔率 - 败率) / 赔率
其中：赔率 = 盈利金额 / 亏损金额

### 金字塔加仓法
初始仓位较小，趋势确认后逐步加仓。
- 第一次：30%仓位试探
- 第二次：趋势明确后加30%
- 第三次：强势确认后加40%

## 止损策略

### 固定止损
买入后设定固定亏损比例（如-8%）止损。
优点：简单明确
缺点：可能被正常波动洗出

### 移动止损
随着价格上涨，逐步提高止损位。
- 价格每上涨10%，止损位上移5%
- 保护已有利润

### 支撑位止损
以关键技术支撑位作为止损位。
优点：符合技术分析逻辑
缺点：支撑位可能被短暂跌破

## 风险收益比

### 计算方法
风险收益比 = 预期盈利 / 预期亏损

### 应用原则
- 风险收益比 < 1:1 不值得参与
- 理想情况 > 3:1
- 每笔交易都应该先计算风险收益比

## 系统性风险防范

### 宏观风险指标
- 市场整体PE/PB水平
- 恐慌指数（VIX）
- 北向资金流向
- 两融余额变化

### 黑天鹅应对
1. 永远不要满仓
2. 跨市场、跨行业分散
3. 保留一定比例现金
4. 设置极端情况下的应对预案
""",
}


def ui_knowledge_base():
    st.title("📚 知识库管理")
    st.caption("上传投资知识文档，AI 将基于知识库回答专业问题")

    # ========== 预置知识导入 ==========
    st.subheader("📥 预置知识")
    st.caption("一键导入投资技术分析、基本面分析、风险管理等基础文档")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("导入全部预置知识", type="primary", width="stretch"):
            total = 0
            for filename, content in PRESET_KNOWLEDGE.items():
                docs = split_text(content, source_name=filename)
                add_documents(docs)
                total += len(docs)
            st.success(f"已导入 {len(PRESET_KNOWLEDGE)} 个文档, 共 {total} 个文本块")
            st.rerun()

    with col2:
        if st.button("查看预置内容", width="stretch"):
            st.session_state.show_presets = not st.session_state.get("show_presets", False)
            st.rerun()

    # 显示预置内容详情
    if st.session_state.get("show_presets", False):
        for filename, content in PRESET_KNOWLEDGE.items():
            with st.expander(f"📄 {filename}"):
                st.text(content[:500] + "..." if len(content) > 500 else content)

    st.divider()

    # ========== 文档上传 ==========
    st.subheader("📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF 和 TXT 格式",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("开始处理并入库", type="primary"):
            progress_bar = st.progress(0)
            total_chunks = 0

            for i, file in enumerate(uploaded_files):
                try:
                    # 加载文档
                    filename, content = load_uploaded_file(file)
                    # 分块
                    docs = split_text(content, source_name=filename)
                    # 入库
                    add_documents(docs)
                    total_chunks += len(docs)
                    st.success(f"✅ {filename}: {len(docs)} 个文本块")
                except Exception as e:
                    st.error(f"❌ {file.name}: {str(e)}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.success(f"处理完成! 共导入 {total_chunks} 个文本块")
            st.rerun()

    st.divider()

    # ========== 已入库文档列表 ==========
    st.subheader("📋 已入库文档")
    docs = list_documents()

    if docs:
        for doc in docs:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"📄 **{doc['source']}**")
            with col2:
                st.caption(f"{doc['chunks']} 块")
            with col3:
                if st.button("删除", key=f"del_{doc['source']}", width="stretch"):
                    delete_document(doc["source"])
                    st.rerun()

        st.caption(f"共 {len(docs)} 个文档, {sum(d['chunks'] for d in docs)} 个文本块")

        # 清空全部
        if st.button("🗑 清空全部知识库", type="secondary"):
            clear_all()
            st.rerun()
    else:
        st.info("知识库为空，请上传文档或导入预置知识")

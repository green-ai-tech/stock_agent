"""
提示词模块
    - 系统提示词：
        1. default:股市通用聊天提示词
        2. primary:基本面分析师提示词
        3. technolongy:技术分析提示词
        4. finance:财务分析提示词
        5. decision:决策分析提示词

"""
from typing import Dict,Optional
from datetime import datetime           #在提示词中，永远放置一个当前时间：提示词固定，使用工具

#======================系统提示词=====================
SYSTEM_PROMPTS: Dict[str, str] = {
    # 通用股票聊天助手提示词（使用DeepSeek工具生成）
    "default": 
        """
角色：你是一位专业股市投资顾问，拥有15年A股、港股、美股市场分析经验，曾任职于顶级券商研究部。你以客观、理性、合规的分析风格著称，帮助用户在波动的市场中做出明智决策。
你的核心信条：敬畏市场，理性投资，风险第一，收益第二。
        
你具备并运用以下专业知识：
1. 技术分析能力
    - 解读K线形态（头肩顶、双底、旗形整理等）
    - 识别关键支撑位/压力位
    - 运用常用指标（MACD、KDJ、RSI、布林带、均线系统）
2. 基本面分析能力
    - 解读财报数据（营收、净利润、毛利率、ROE）
    - 评估估值指标（PE、PB、PEG、PS）
    - 分析行业趋势和公司竞争优势
3. 市场情绪判断
    - 识别恐慌/贪婪极端时刻
    - 解读资金流向和成交量变化
    - 跟踪北向资金、两融余额等关键数据
4. 风险管理能力
    - 计算仓位管理建议
    - 识别潜在止损位
    - 评估风险收益比
5. 信息整合能力
    - 理解宏观政策影响
    - 跟踪重要经济数据（CPI、PMI、GDP）
    - 解读突发事件的市场影响

你的行为准则（严格遵守）
1. **必须做到的**：
    - 先问后答：在给出具体建议前，主动询问用户：
        - 投资目标（短线/中线/长线）
        - 风险承受能力（保守/稳健/进取）
        - 当前仓位情况
        - 计划投资金额
    - 风险提示前置：每次给出操作建议前，必须附带清晰的风险声明
    - 给出明确依据：每个判断必须列出至少2个分析理由
    - 承认不确定性：明确说出”我不确定“或”这需要更多信息“，绝不编造数据
    - 推荐可靠信源：引导用户关注官方披露、交易所公告、权威财经媒体
2. **绝对禁止的**：
    - 不承诺收益：绝不说”肯定涨“、”保证赚“、”稳赚不赔“
    - 不推荐具体买卖点：不给出精确的买入/卖出价格指令
    - 不预测极端行情：不对”牛市何时结束“、”这波能涨到多少“等给出确定性预测
    - 不贬低其他投资标的：不恶意批评用户持有的股票或其他投资品种
    - 不越界提供其他专业建议：不提供法律、税务、遗产规划等非投资领域的建议
    - 不追逐热点炒作：不对”妖股“、”内幕消息“类问题提供分析

你的回答风格：
1. 格式规范：
    -【核心结论】：（1-2句话概括）
    -【分析依据】：
        - 技术面理由
        - 基本面
        - 新闻
        - 财务状况

    -【风险提示】：
        - 关键风险点

    -【操作建议】：
        - 仓位/方向建议，非具体买卖点
    -【补充说明】：（可选，延伸知识点或需要用户补充的信息）
2. 语气要求：
    - 冷静克制：不使用”暴涨“、”暴跌“、”疯狂“、”惊人“等情绪化词汇
    - 专业平实：使用市场通用术语，但需对专业名词做简要解释
    - 不卑不亢：既不过度自信，也不过度保守
    - 
3. 解释风格
    - 预测和回答可能的疑问
    - 确保逻辑连贯    

当前时间：{current_time}\n\n

请根据用户的问题，提供有价值的帮助，如果需要绘制K线图等图表，请使用K线图绘制工具。
    """,
# 补充，系统提示词中如果提到了使用工具，需要补充说明工具使用的时机， 以及上下文记忆，以及使用提示。

    # 基本分析师提示词
    "primary": 
        """
        """,

    # 技术分析师提示词
    "technology": 
        """
        """,

    # 财务分析师提示词
    "finance": 
        """
        """,

    # 决策分析师提示词
    "decision": 
        """
        """,
}




#======================提示词函数=====================
def get_system_prompt(
    mode: str = "default",
    custom_instructions: Optional[str] = None,
    include_time: bool = True,
) -> str:
    """
    获取系统提示词
    
    根据不同的提示词模式返回相应的系统提示词，支持自定义补充说明。
    
    Args:
        mode: 提示词模式，可选值见 SYSTEM_PROMPTS 的键
        custom_instructions: 自定义补充说明，会追加到系统提示词后
        include_time: 是否包含当前时间
        
    Returns:
        系统提示词
        
    Raises:
        ValueError: 如果模式不存在
    """
    if mode not in SYSTEM_PROMPTS:
        available_modes = ", ".join(SYSTEM_PROMPTS.keys())
        raise ValueError(f"未知的提示词模式: {mode}. 可用模式: {available_modes}")
    
    # 获取基础提示词
    prompt = SYSTEM_PROMPTS[mode]
    
    # 插入当前日期与时间（也可以采用提示词模版）
    if include_time:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = prompt.format(current_time=current_time)       #提示词占位符，格式化为时间，
    else:
        # 如果不包含时间，移除时间占位符
        prompt = prompt.replace("当前时间：{current_time}\n\n", "")#如果不包含时间，替换为“” 空
    
    # 添加自定义说明
    if custom_instructions:
        prompt += f"\n\n补充说明：\n{custom_instructions}"
    
    return prompt


# 定制提示词模版
def create_custom_prompt(
    role: str,
    capabilities: list[str],
    principles: list[str],
    additional_context: Optional[str] = None,
) -> str:
    """
    创建自定义系统提示词
    提供一个标准的系统提示词结构
    Args:
        role: AI 的角色描述
        capabilities: 能力列表
        principles: 行为准则列表
        additional_context: 额外的上下文信息
        
    Returns:
        构建好的系统提示词
        
    使用例子:       # 全部采用MD格式（markdown）
        >>> prompt = create_custom_prompt(
        ...     role="数学学习助手",                                #角色
        ...     capabilities=["解答数学问题", "讲解数学概念"],        #能力
        ...     principles=["循序渐进", "注重理解"],                 #准则
        ...     additional_context="用户正在准备高考"                #场景
        ... )
    """
    prompt_parts = [f"你是 {role}。"]
    
    # 添加能力描述
    if capabilities:
        prompt_parts.append("\n你的能力：")
        for i, cap in enumerate(capabilities, 1):
            prompt_parts.append(f"{i}. {cap}")
    
    # 添加行为准则
    if principles:
        prompt_parts.append("\n你的准则：")
        for principle in principles:
            prompt_parts.append(f"- {principle}")
    
    # 添加额外上下文
    if additional_context:
        prompt_parts.append(f"\n{additional_context}")
    
    # 添加当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt_parts.append(f"\n当前时间：{current_time}")
    
    return "\n".join(prompt_parts)

#提示词中工具是变化的：如何让提示词随着工具的改变而改变？
TOOL_USAGE_INSTRUCTIONS = """
可用工具说明：
    - get_current_time: 获取当前时间和日期（仅在需要精确时间戳时使用）
    - get_current_date：获取当前时间和星期（仅在需要知道星期几的时候使用）
使用工具的时机：
    - 需要知道当前时间或日期时，使用 get_current_time（注意：查询天气时不需要先调用此工具）
    - 需要知道当前是星期几时，使用 get_current_date
"""


def get_prompt_with_tools(mode: str = "default") -> str:        #这里工具提示词已经固定（建议留一个参数：由用户指定）
    """
    获取包含工具使用说明的系统提示词
    
    Args:
        mode: 基础提示词模式
        
    Returns:
        包含工具说明的完整提示词
    """
    base_prompt = get_system_prompt(mode)
    return f"{base_prompt}\n\n{TOOL_USAGE_INSTRUCTIONS}"
#======================


if __name__ == "__main__":
    prompt = get_prompt_with_tools("default")
    print(prompt)

"""
    Agent的封装（面向对象：数据agent与函数invoke，stream）
        - BaseAgent
            - invoke/ainvke
            - stream/astream
"""


from typing import List,Optional,Dict,Any,Iterator,AsyncIterator,Union,Sequence         #约束定义类型
from langchain.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver   #短期记忆 MemoSaver（老版本）


from utils import get_logger
from .models import get_chat_model
from .tools import BASIC_TOOLS
from .prompts import SYSTEM_PROMPTS, get_prompt_with_tools, get_system_prompt, create_custom_prompt


logger = get_logger(__name__)
#封装BaseAgent
class BaseAgent:
    """
    langchain的 create_agent 函数的封装类。
        1. 提供短期记忆
        2. 上下文聊天历史管理
        3. 基本的工具
        4. 提供RAG
        5. ......（历史数据自动长期存储）
    
    Attributes:
        model: LLM 模型实例或模型标识符
        tools: Agent 可用的工具列表
        system_prompt: 系统提示词
        
        graph: LangChain 的 CompiledStateGraph 实例（由 create_agent 返回）
    
    """
    def __init__(
        self,
        model: Optional[Union[str, BaseChatModel]] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        system_prompt: Optional[str] = None, #system_prompt与prompt_mode，二选一，
        prompt_mode: str = "default",
        **kwargs: Any,
    ):
        """
        初始化 Base Agent
        Args:
            model: LLM 模型，可以是：
                - 字符串标识符（如 "ollama:gemma4:e4b"）
                - BaseChatModel 实例
                - None，使用默认配置创建
            tools: Agent 可用的工具列表, 可以是：
                - Sequence[BaseTool]
                - None 或 空列表
            system_prompt: 自定义系统提示词，可以是：
                - str
                - 如果为 None，则根据 prompt_mode 生成
            prompt_mode: 提示词模式
                - default：通用股票聊天助手
                - primary：基本分析师
                - technology：技术分析师
                - finance：财务分析师
                - decision：决策分析师
            **kwargs: 其他传递给 create_agent 的参数，如：
                - store: 跨线程数据存储
                - state_schema：状态模式
                - context_schema:上下文模式
                - ...等create_agent参数
        """
        # ==================== 模型初始化 ====================
        if model is None:
            
            # 使用get_chat_model加载默认模型以及默认配置
            self.model = get_chat_model()  #默认：来自配置文件
            logger.info(f"使用默认模型: {self.model}")
        elif isinstance(model, str):
            # 字符串标识符
            self.model = model
            logger.info(f"使用模型: {model}")
        else:
            # BaseChatModel 实例
            self.model = model
            logger.info(f"使用自定义模型: {model.__class__.__name__}")
        
        # ==================== 工具初始化 ====================
        if tools is None:
            self.tools = BASIC_TOOLS
        else:
            self.tools = list(tools) if tools else []
            self.tools.extend(BASIC_TOOLS)

        if self.tools and isinstance(self.model, BaseChatModel):
            try:
                self.model.bind_tools(self.tools)
                logger.info(f"使用的工具集 ({len(self.tools)} 个工具)")
            except NotImplementedError:
                logger.warning("当前模型不支持 bind_tools，已自动禁用工具")
                self.tools = []
            except Exception as e:
                logger.warning(f"工具能力检测失败，已自动禁用工具: {repr(e)}")
                self.tools = []
        else:
            logger.info(f"使用的工具集 ({len(self.tools)} 个工具)")

        # ==================== 提示词初始化 ====================
        if system_prompt is None:
            # 根据模式生成系统提示词
            if self.tools:
                # 如果有工具，使用包含工具说明的提示词
                self.system_prompt = get_prompt_with_tools(mode=prompt_mode)
                logger.info(f"使用的系统提示词 (模式: {prompt_mode})")
            else:
                # 没有工具，使用普通提示词
                self.system_prompt = get_system_prompt(mode=prompt_mode)
                logger.info(f"使用的系统提示词 (模式: {prompt_mode})")
        else:
            self.system_prompt = system_prompt
            logger.info("使用自定义系统提示词")
            
        # ==================== 短期记忆 ====================
        self.checkpointer = InMemorySaver()  # 记得在invoke中配置config
        logger.info(f"使用短期记忆 (InMemorySaver)")
        # 其他参数的使用，可以全部初始化response_format，state_schema，context_schema，store
        # ==================== 创建 Agent ====================
        try:
            logger.info("创建 Agent ...")
            # 调用 create_agent
            self.graph = create_agent(
                model=self.model,
                tools=self.tools if self.tools else None,  # None 或空列表表示无工具
                system_prompt=self.system_prompt,
                checkpointer=self.checkpointer,
                **kwargs,  # 支持 checkpointer, store, interrupt_before/after, name 等
            )
            
            logger.success("Agent 创建成功（CompiledStateGraph）")
            logger.success(f"\t配置: tools={len(self.tools)}")
 
        except Exception as e:
            logger.error(f"Agent 创建失败: {e}")
            raise
    
    def invoke(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        thread_id: str = "public_001",
        **kwargs: Any,
    ) -> str:
        """
        非流式同步调用
        Args:
            input_text: 用户输入的文本，自动转换为{"messages":[HumanMessage(....)]}
            chat_history: 对话历史（可选）
            **kwargs: 其他传递给 graph 的参数
        Returns:
            Agent 的响应文本

        """
        logger.info(f"执行 Agent 调用: {input_text[:50]}...")
        
        try:
            messages = []
            # 添加历史消息
            if chat_history:
                messages.extend(chat_history)
            
            # 添加当前用户消息
            messages.append(HumanMessage(content=input_text))
            
            # 准备输入
            graph_input = {"messages": messages}
            graph_input.update(kwargs)
            
            # 执行 Graph
            # CompiledStateGraph 的 invoke 方法返回最终状态
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            result = self.graph.invoke(graph_input, config=config)
            
            # 提取最后一条 AI 消息
            # result 是一个包含 "messages" 键的字典
            output_messages = result.get("messages", [])
            
            # 找到最后一条 AI 消息
            ai_response = ""
            for msg in reversed(output_messages):
                if isinstance(msg, AIMessage):
                    ai_response = msg.content
                    break
            
            logger.success(f"Agent 调用完成，输出长度: {len(ai_response)} 字符")
            # logger.success(f"\t输出: {ai_response[:100]}...")
            
            return ai_response
            
        except Exception as e:
            error_msg = f"Agent 执行失败: {str(e)}"
            logger.error(f"{error_msg}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    def stream(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        stream_mode: str = "messages",
        thread_id: str = "public_001",
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        流式调用 Agent
        
        Args:
            input_text: 用户输入的文本
            chat_history: 对话历史（可选）
            stream_mode: 流式模式，可选值：
                        - "messages": 流式返回消息内容（推荐）
                        - "updates": 返回状态更新
                        - "values": 返回完整状态值
            **kwargs: 其他参数
            
        Yields:
            Agent 输出的文本片段(使用生成器返回)
            
        """
        logger.info(f"执行 Agent 流式调用: {input_text[:50]}...")
        
        try:
            # 准备消息列表
            messages = []
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=input_text))
            
            # 准备输入
            graph_input = {"messages": messages}
            graph_input.update(kwargs)
            
            # 流式执行 Graph
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            # CompiledStateGraph 的 stream 方法支持多种模式  （stream_mode对stream函数无效）
            for chunk in self.graph.stream(graph_input, stream_mode=stream_mode, config=config):
                # 根据 stream_mode 处理不同的输出格式
                if stream_mode == "messages":
                    # messages 模式：chunk 是 (message, metadata) 元组
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        message, metadata = chunk
                        if isinstance(message, (AIMessage, AIMessageChunk)) and message.content:
                            # logger.success(f"\t流式输出: {message.content[:50]}...")
                            yield message.content   #没使用return，使用yield（协程）
                    elif isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                        # logger.success(f"\t流式输出: {chunk.content[:50]}...")
                        yield chunk.content
                
                elif stream_mode == "updates":
                    # updates 模式：chunk 是状态更新字典
                    if isinstance(chunk, dict) and "messages" in chunk:
                        messages_update = chunk["messages"]
                        if messages_update:
                            last_msg = messages_update[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                yield last_msg.content
            
            logger.success("Agent 流式调用完成")
            
        except Exception as e:
            if isinstance(e, NotImplementedError):
                logger.warning("当前模型或适配器不支持 stream，已自动回退为 invoke")
                fallback_response = self.invoke(
                    input_text=input_text,
                    chat_history=chat_history,
                    thread_id=thread_id,
                    **kwargs,
                )
                if fallback_response:
                    yield fallback_response
                return

            error_detail = repr(e)
            logger.exception(f"Agent 流式执行失败: {error_detail}")
            yield f"\n\n抱歉，处理您的请求时出现错误: {error_detail}"

PRESET_CONFIGS = {
    # ... 其他预设 ...
    "structure": {
        "model_name": "ollama:qwen3.5:9b",
        "temperature": 0.0,
        "description": "结构化输出需要准确稳定的模型",
    },
}

#工厂生成对象
def create_base_agent(
    model: Optional[Union[str, BaseChatModel]] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    prompt_mode: str = "default",
    **kwargs: Any,
) -> BaseAgent:
    """
    创建基础 Agent 的便捷工厂函数
    Args:
        model: LLM 模型（字符串标识符或实例）
        tools: 工具列表
        prompt_mode: 提示词模式
        **kwargs: 其他参数（传递给 create_agent）        
    Returns:
        配置好的 BaseAgent 实例
    """
    logger.info(f"创建 Base Agent (mode={prompt_mode})")
    
    return BaseAgent(   #二传（封装者，最理解这个类）
        model=model,
        tools=tools,
        prompt_mode=prompt_mode,
        **kwargs,
    )
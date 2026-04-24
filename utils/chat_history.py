"""
会话历史持久化服务
- 会话管理：创建、查询、重命名、删除
- 消息管理：追加、查询
"""
from typing import List, Optional, Dict
from utils.db import get_db_session, Conversation, Message
from utils.logger import logger


def create_conversation(user_id: int, agent_type: str = "base", title: str = None) -> Optional[Conversation]:
    """创建新会话"""
    session = get_db_session()
    try:
        conv = Conversation(
            user_id=user_id,
            title=title or "新对话",
            agent_type=agent_type,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)
        logger.info(f"创建会话成功: id={conv.id}, type={agent_type}")
        return conv
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        session.rollback()
        return None
    finally:
        session.close()


def get_user_conversations(user_id: int, agent_type: str = None) -> List[Conversation]:
    """获取用户的所有会话，按更新时间倒序"""
    session = get_db_session()
    try:
        query = session.query(Conversation).filter(Conversation.user_id == user_id)
        if agent_type:
            query = query.filter(Conversation.agent_type == agent_type)
        return query.order_by(Conversation.updated_at.desc()).all()
    finally:
        session.close()


def get_conversation(conv_id: int) -> Optional[Conversation]:
    """获取单个会话"""
    session = get_db_session()
    try:
        return session.query(Conversation).filter(Conversation.id == conv_id).first()
    finally:
        session.close()


def rename_conversation(conv_id: int, new_title: str) -> bool:
    """重命名会话"""
    session = get_db_session()
    try:
        conv = session.query(Conversation).filter(Conversation.id == conv_id).first()
        if not conv:
            return False
        conv.title = new_title[:200]
        session.commit()
        return True
    except Exception as e:
        logger.error(f"重命名会话失败: {e}")
        session.rollback()
        return False
    finally:
        session.close()


def delete_conversation(conv_id: int) -> bool:
    """删除会话及其所有消息"""
    session = get_db_session()
    try:
        conv = session.query(Conversation).filter(Conversation.id == conv_id).first()
        if not conv:
            return False
        session.delete(conv)
        session.commit()
        logger.info(f"删除会话: id={conv_id}")
        return True
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        session.rollback()
        return False
    finally:
        session.close()


def add_message(conv_id: int, role: str, content: str, chart_paths: Dict = None) -> Optional[Message]:
    """向会话中追加一条消息"""
    session = get_db_session()
    try:
        msg = Message(
            conversation_id=conv_id,
            role=role,
            content=content,
            chart_paths=chart_paths,
        )
        session.add(msg)

        # 更新会话的 updated_at
        conv = session.query(Conversation).filter(Conversation.id == conv_id).first()
        if conv:
            conv.updated_at = msg.created_at

        session.commit()
        session.refresh(msg)
        return msg
    except Exception as e:
        logger.error(f"添加消息失败: {e}")
        session.rollback()
        return None
    finally:
        session.close()


def get_messages(conv_id: int) -> List[Message]:
    """获取会话的所有消息，按 id 正序"""
    session = get_db_session()
    try:
        return session.query(Message).filter(
            Message.conversation_id == conv_id
        ).order_by(Message.id.asc()).all()
    finally:
        session.close()


def generate_title(first_message: str) -> str:
    """从用户第一条消息生成会话标题（取前20字）"""
    text = first_message.strip()
    if len(text) <= 20:
        return text
    return text[:20] + "..."

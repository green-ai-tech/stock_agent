import bcrypt
from utils.db import get_db_session, User
from utils.logger import logger
from datetime import datetime

def hash_password(plain_password: str) -> str:
    """对明文密码进行 bcrypt 哈希"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码与哈希是否匹配"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def check_login(username: str, password: str) -> bool:
    """
    从数据库查询用户并验证密码
    返回 True 表示验证成功，False 表示失败
    同时更新 last_login 时间（可选）
    """
    session = get_db_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            logger.warning(f"登录失败：用户 {username} 不存在")
            return False
        
        if verify_password(password, user.password_hash):
            # 更新最后登录时间
            user.last_login = datetime.utcnow()
            session.commit()
            logger.info(f"用户 {username} 登录成功")
            return True
        else:
            logger.warning(f"登录失败：用户 {username} 密码错误")
            return False
    except Exception as e:
        logger.error(f"登录验证数据库异常: {e}")
        return False
    finally:
        session.close()

def get_user_by_username(username: str):
    """根据用户名获取用户对象，不存在返回 None"""
    session = get_db_session()
    try:
        return session.query(User).filter(User.username == username).first()
    finally:
        session.close()

def create_user(username: str, password: str, email: str = None) -> bool:
    """
    创建新用户
    返回 True 表示成功，False 表示用户名已存在
    """
    session = get_db_session()
    try:
        # 检查用户名是否已存在
        existing = session.query(User).filter(User.username == username).first()
        if existing:
            logger.warning(f"注册失败：用户名 {username} 已存在")
            return False
        
        hashed_pw = hash_password(password)
        new_user = User(
            username=username,
            password_hash=hashed_pw,
            email=email
        )
        session.add(new_user)
        session.commit()
        logger.info(f"新用户注册成功：{username}")
        return True
    except Exception as e:
        logger.error(f"注册异常: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def update_password(username: str, new_password: str) -> bool:
    """
    更新用户密码（管理员或用户自己调用）
    返回 True 表示成功，False 表示用户不存在
    """
    session = get_db_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if not user:
            logger.warning(f"修改密码失败：用户 {username} 不存在")
            return False
        
        user.password_hash = hash_password(new_password)
        session.commit()
        logger.info(f"用户 {username} 密码已更新")
        return True
    except Exception as e:
        logger.error(f"修改密码异常: {e}")
        session.rollback()
        return False
    finally:
        session.close()
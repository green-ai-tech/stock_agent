import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import create_tables, get_db_session, User, Conversation, Message
from utils.auth import hash_password

def init_database():
    print("创建数据库表...")
    create_tables()

    # 验证表是否存在
    session = get_db_session()
    try:
        session.execute(User.__table__.select().limit(0))
        print("  ✅ users 表")
        session.execute(Conversation.__table__.select().limit(0))
        print("  ✅ conversations 表")
        session.execute(Message.__table__.select().limit(0))
        print("  ✅ messages 表")
    except Exception as e:
        print(f"  ❌ 表验证失败: {e}")
    finally:
        session.close()

    print("创建测试用户（admin / 123456）...")
    session = get_db_session()
    try:
        existing = session.query(User).filter(User.username == "admin").first()
        if not existing:
            test_user = User(
                username="admin",
                password_hash=hash_password("123456"),
                email="admin@example.com"
            )
            session.add(test_user)
            session.commit()
            print("测试用户创建成功: 用户名=admin, 密码=123456")
        else:
            print("用户 admin 已存在，跳过创建")
    except Exception as e:
        print(f"初始化失败: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    init_database()

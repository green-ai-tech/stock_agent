import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import create_tables, get_db_session, User
from utils.auth import hash_password

def init_database():
    print("创建数据库表...")
    create_tables()
    
    print("创建测试用户（admin / 123456）...")
    session = get_db_session()
    try:
        # 检查是否已存在 admin 用户
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
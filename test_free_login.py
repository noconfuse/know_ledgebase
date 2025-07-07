#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试free_login接口的脚本
"""

import base64
import json
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from config import settings

def _create_cipher_suite():
    """创建加密套件"""
    # 使用FREE_LOGIN_TOKEN作为密钥生成Fernet密钥
    password = settings.FREE_LOGIN_TOKEN.encode()
    salt = b'salt_1234567890'  # 固定盐值，与后端保持一致
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return Fernet(key)

def create_test_token(unique_id: str) -> str:
    """创建测试用的加密token"""
    # 创建payload
    payload = {
        "unique_id": unique_id
    }
    
    # 转换为JSON字符串
    json_data = json.dumps(payload)
    
    # 加密
    cipher_suite = _create_cipher_suite()
    encrypted_data = cipher_suite.encrypt(json_data.encode())
    
    # 编码为base64字符串
    token = base64.urlsafe_b64encode(encrypted_data).decode()
    
    return token

def test_free_login(unique_id: str, base_url: str = "http://localhost:8001"):
    """测试free_login接口"""
    # 1. 创建测试token
    test_token = create_test_token(unique_id)
    print(f"Generated test token for unique_id '{unique_id}': {test_token}")
    
    # 2. 调用free_login接口
    url = f"{base_url}/auth/free_login"
    payload = {
        "token": test_token
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nResponse status: {response.status_code}")
        print(f"Response body: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                access_token = data["data"]["access_token"]
                user_info = data["data"]["user"]
                print(f"\n✅ 登录成功!")
                print(f"用户名: {user_info['username']}")
                print(f"用户ID: {user_info['id']}")
                print(f"访问令牌: {access_token[:50]}...")
                return access_token
            else:
                print(f"\n❌ 登录失败: {data.get('message')}")
        else:
            print(f"\n❌ 请求失败: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 网络请求错误: {e}")
    except Exception as e:
        print(f"\n❌ 其他错误: {e}")
    
    return None

def test_with_access_token(access_token: str, base_url: str = "http://localhost:8001"):
    """使用访问令牌测试获取用户信息"""
    url = f"{base_url}/auth/me"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"\n获取用户信息 - 状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"\n❌ 获取用户信息失败: {e}")

if __name__ == "__main__":
    # 测试不同的unique_id
    test_cases = [
        "user_12345",
        "test_user_001", 
        "mobile_user_abc"
    ]
    
    print("🚀 开始测试 free_login 接口...\n")
    
    for unique_id in test_cases:
        print(f"\n{'='*50}")
        print(f"测试 unique_id: {unique_id}")
        print(f"{'='*50}")
        
        # 第一次调用 - 应该创建新用户
        print("\n第一次调用 (创建用户):")
        access_token = test_free_login(unique_id)
        
        if access_token:
            # 测试使用访问令牌获取用户信息
            test_with_access_token(access_token)
            
            # 第二次调用 - 应该直接登录现有用户
            print("\n第二次调用 (登录现有用户):")
            test_free_login(unique_id)
    
    print("\n🎉 测试完成!")
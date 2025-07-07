#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import hashlib
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def decrypt_cryptojs_token(encrypted_token: str, password: str) -> dict:
    """
    解密CryptoJS AES加密的token
    
    前端加密流程：
    1. JSON.stringify(payload)
    2. CryptoJS.AES.encrypt(jsonString, freeLoginToken).toString()
    3. CryptoJS.enc.Base64.stringify(CryptoJS.enc.Utf8.parse(encrypted))
    """
    try:
        print(f"\n=== 开始解密 ===")
        print(f"密钥: {password}")
        print(f"Token长度: {len(encrypted_token)}")
        
        # 步骤1: 第一层Base64解码
        print("\n步骤1: 第一层Base64解码")
        first_decode = base64.b64decode(encrypted_token).decode('utf-8')
        print(f"第一层解码结果: {first_decode[:100]}...")
        
        # 步骤2: 第二层Base64解码（CryptoJS AES加密结果）
        print("\n步骤2: 第二层Base64解码")
        encrypted_data = base64.b64decode(first_decode)
        print(f"第二层解码长度: {len(encrypted_data)}")
        print(f"前16字节: {encrypted_data[:16]}")
        
        # 步骤3: 检查CryptoJS格式
        print("\n步骤3: 检查CryptoJS格式")
        if encrypted_data[:8] != b'Salted__':
            raise ValueError("不是有效的CryptoJS加密格式")
        
        # 提取盐值和密文
        salt = encrypted_data[8:16]
        ciphertext = encrypted_data[16:]
        print(f"盐值: {salt.hex()}")
        print(f"密文长度: {len(ciphertext)}")
        
        # 步骤4: 生成密钥和IV（使用CryptoJS兼容的方法）
        print("\n步骤4: 生成密钥和IV")
        password_bytes = password.encode('utf-8')
        
        # CryptoJS使用EVP_BytesToKey算法
        # 等价于: MD5(password + salt) + MD5(MD5(password + salt) + password + salt)
        def evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int) -> tuple:
            """CryptoJS兼容的密钥派生函数"""
            d = d_i = b''
            while len(d) < (key_len + iv_len):
                d_i = hashlib.md5(d_i + password + salt).digest()
                d += d_i
            return d[:key_len], d[key_len:key_len+iv_len]
        
        key, iv = evp_bytes_to_key(password_bytes, salt, 32, 16)
        print(f"生成的密钥: {key.hex()[:32]}...")
        print(f"生成的IV: {iv.hex()}")
        
        # 步骤5: AES解密
        print("\n步骤5: AES解密")
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(ciphertext)
        print(f"解密后数据长度: {len(decrypted_padded)}")
        
        # 步骤6: 去除PKCS7填充
        print("\n步骤6: 去除填充")
        try:
            decrypted_data = unpad(decrypted_padded, AES.block_size)
        except ValueError as e:
            print(f"标准去填充失败: {e}")
            # 手动去除填充
            padding_length = decrypted_padded[-1]
            if padding_length <= AES.block_size:
                decrypted_data = decrypted_padded[:-padding_length]
                print(f"手动去除填充: {padding_length}字节")
            else:
                decrypted_data = decrypted_padded
                print("无法去除填充，使用原始数据")
        
        # 步骤7: 解码为字符串
        print("\n步骤7: 解码为字符串")
        try:
            json_string = decrypted_data.decode('utf-8')
            print(f"解密后的JSON: {json_string}")
        except UnicodeDecodeError as e:
            print(f"UTF-8解码失败: {e}")
            print(f"十六进制数据: {decrypted_data.hex()[:200]}...")
            return None
        
        # 步骤8: 解析JSON
        print("\n步骤8: 解析JSON")
        try:
            payload = json.loads(json_string)
            print(f"解析成功: {payload}")
            return payload
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return None
            
    except Exception as e:
        print(f"解密失败: {e}")
        return None

if __name__ == "__main__":
    # 测试token
    test_token = "VTJGc2RHVmtYMSs3aW5MTGJ0cElLdlROZWlWU1JDRmY0N0NCcVZqWGhzcVk1cDZ4eXozNzFsRzkxSW9kNlh5M216NFQxeFhFVzM1MThPQ1lyY3dMRjVlVnlPbGkxM0xBeW5SM2dEN3dHNllQT3dPWnZSRDcwK0dkdzlJQjhta1ltSi8reGZpUkNWVWdvTUQwZGNOcHNDaENnUjZmd2w4TVZ0SmEvRzNUSk1ERXoybkdlM05CYmh5NHNaUTU5K0JseVEvcnpQMEV6VERDOXB6TFVtMG1Pbm4wTVFpWWg3a252THFSWStHZlVqRUkzZWVIS3ovbFBEeTN1RHM0UWlUcjZjMnZSNkVSVzZhUW43M3F4L0k2V2NtekpybytSKzdPejFhM2NuWUM0ZXdJOEZDZG5kcnhRUWZSR3UzaDU3eVo="
    
    # 使用正确的密钥
    password = "free-login-token-please-change-in-production"
    
    print("开始CryptoJS兼容解密测试...")
    result = decrypt_cryptojs_token(test_token, password)
    
    if result:
        print(f"\n✅ 解密成功！")
        print(f"Payload: {result}")
        if 'unique_id' in result:
            print(f"unique_id: {result['unique_id']}")
        if 'timestamp' in result:
            print(f"timestamp: {result['timestamp']}")
    else:
        print("\n❌ 解密失败")
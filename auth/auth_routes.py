from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
import base64
import json
from passlib.context import CryptContext
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import hashlib

from models.database import get_db
from dao.user_dao import UserDAO
from .schemas import UserCreate, UserLogin, UserResponse, Token, UserUpdate, FreeLoginRequest
from .dependencies import get_current_user, get_current_active_user
from config import settings
from common.response import success_response, error_response, ErrorCodes
from common.exception_handler import setup_exception_handlers

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    """创建刷新令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

@router.post("/register")
async def register(user_data: UserCreate, db: Session = Depends(get_db)) -> JSONResponse:
    """用户注册"""
    try:
        # 检查用户是否已存在
        existing_user = UserDAO.get_user_by_username(db, user_data.username)
        if existing_user:
            return error_response(
                message="用户名已被注册",
                error_code=ErrorCodes.USER_ALREADY_EXISTS,
                status_code=400
            )
        
        if user_data.email:
            existing_email = UserDAO.get_user_by_email(db, user_data.email)
            if existing_email:
                return error_response(
                    message="邮箱已被注册",
                    error_code=ErrorCodes.EMAIL_ALREADY_EXISTS,
                    status_code=400
                )
        
        # 创建用户
        user = UserDAO.create_user(db, user_data)
        if not user:
            return error_response(
                message="创建用户失败",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=500
            )
        
        user_response = UserResponse.model_validate(user)
        return success_response(
            data=user_response.model_dump(),
            message="用户注册成功",
            status_code=201
        )
    except Exception as e:
        return error_response(
            message="注册过程中发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@router.post("/login")
async def login(user_data: UserLogin, db: Session = Depends(get_db)) -> JSONResponse:
    """用户登录"""
    try:
        # 验证用户
        user = UserDAO.authenticate_user(db, user_data.username, user_data.password)
        if not user:
            return error_response(
                message="用户名或密码错误",
                error_code=ErrorCodes.INVALID_CREDENTIALS,
                status_code=401
            )
        
        if not user.is_active:
            return error_response(
                message="用户账户已被禁用",
                error_code=ErrorCodes.INACTIVE_USER,
                status_code=400
            )
        
        # 创建令牌
        access_token = create_access_token(data={"sub": user.username, "user_id": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": user.username, "user_id": str(user.id)})
        
        # 保存会话令牌
        UserDAO.create_user_session(db, user.id, refresh_token)
        
        # 更新最后登录时间
        UserDAO.update_last_login(db, user.id)
        
        user_response = UserResponse.model_validate(user)
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600,
            "user": user_response.model_dump()
        }
        
        return success_response(
            data=token_data,
            message="登录成功"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return error_response(
            message="登录过程中发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """刷新访问令牌"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        token_type: str = payload.get("type")
        
        if username is None or user_id is None or token_type != "refresh":
            return error_response(
                message="无效的刷新令牌",
                error_code=ErrorCodes.INVALID_TOKEN,
                status_code=status.HTTP_401_UNAUTHORIZED
            )
            
    except jwt.PyJWTError:
        return error_response(
            message="无效的刷新令牌",
            error_code=ErrorCodes.INVALID_TOKEN,
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    # 检查令牌是否被撤销
    if UserDAO.is_token_revoked(db, token):
        return error_response(
            message="刷新令牌已被撤销",
            error_code=ErrorCodes.TOKEN_REVOKED,
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    # 获取用户
    user = UserDAO.get_user_by_username(db, username)
    if user is None or not user.is_active:
        return error_response(
            message="用户不存在或已被禁用",
            error_code=ErrorCodes.USER_NOT_FOUND,
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    # 创建新的访问令牌
    access_token = create_access_token(data={"sub": user.username, "user_id": str(user.id)})
    return success_response(
        data={
            "access_token": access_token,
            "refresh_token": token,  # 返回原刷新令牌
            "token_type": "bearer"
        },
        message="刷新成功"
    )

@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """用户登出"""
    token = credentials.credentials
    
    # 撤销令牌
    UserDAO.revoke_token(db, token)
    return success_response(
        message="登出成功"
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_active_user)
) -> JSONResponse:
    """获取当前用户信息"""
    return success_response(
        data=current_user,
        message="获取当前用户信息成功"
    )

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """更新当前用户信息"""
    user = UserDAO.update_user(db, current_user["id"], user_update)
    if not user:
        return error_response(
            message="更新用户信息失败",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=status.HTTP_404_NOT_FOUND
        )
    return success_response(
        data=UserResponse.model_validate(user),
        message="更新用户信息成功"
    )

@router.delete("/me")
async def delete_current_user(
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> JSONResponse:
    """删除当前用户账户"""
    success = UserDAO.delete_user(db, current_user["id"])
    if not success:
        return error_response(
            message="删除用户账户失败",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=status.HTTP_404_NOT_FOUND
        )
       
    return success_response(
        message="删除用户账户成功"
    )


def decrypt_token(encrypted_token: str) -> str:
    """解密token获取唯一ID（兼容前端CryptoJS AES加密）"""
    try:
        # 前端加密流程：
        # 1. JSON.stringify(payload)
        # 2. CryptoJS.AES.encrypt(jsonString, freeLoginToken).toString()
        # 3. CryptoJS.enc.Base64.stringify(CryptoJS.enc.Utf8.parse(encrypted))
        
        # 后端解密流程：
        # 1. 从最外层base64解码得到AES加密的字符串
        encrypted_aes_string = base64.b64decode(encrypted_token).decode('utf-8')
        
        # 2. 解析CryptoJS的AES加密格式
        password = settings.FREE_LOGIN_TOKEN.encode('utf-8')
        
        # CryptoJS的AES加密结果是base64格式，需要再次解码
        encrypted_data = base64.b64decode(encrypted_aes_string)
        
        # CryptoJS在加密时会在前8字节添加"Salted__"标识，接下来8字节是盐值
        if encrypted_data[:8] == b'Salted__':
            salt = encrypted_data[8:16]
            ciphertext = encrypted_data[16:]
            
            print(f"调试信息 - 盐值: {salt.hex()}")
            print(f"调试信息 - 密文长度: {len(ciphertext)}")
            
            # 使用CryptoJS兼容的EVP_BytesToKey算法生成密钥和IV
            def evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int) -> tuple:
                """CryptoJS兼容的密钥派生函数"""
                d = d_i = b''
                while len(d) < (key_len + iv_len):
                    d_i = hashlib.md5(d_i + password + salt).digest()
                    d += d_i
                return d[:key_len], d[key_len:key_len+iv_len]
            
            key, iv = evp_bytes_to_key(password, salt, 32, 16)
            
            print(f"调试信息 - 生成的密钥: {key.hex()[:32]}...")
            print(f"调试信息 - 生成的IV: {iv.hex()}")
            
            # AES解密
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted_padded = cipher.decrypt(ciphertext)
            
            # 去除PKCS7填充
            try:
                decrypted_data = unpad(decrypted_padded, AES.block_size)
            except ValueError:
                # 如果标准去填充失败，尝试手动去除填充
                padding_length = decrypted_padded[-1]
                if padding_length <= AES.block_size:
                    decrypted_data = decrypted_padded[:-padding_length]
                else:
                    decrypted_data = decrypted_padded
            
            # 解码为字符串
            try:
                json_string = decrypted_data.decode('utf-8')
            except UnicodeDecodeError:
                print(f"调试信息 - UTF-8解码失败，十六进制数据: {decrypted_data.hex()[:100]}...")
                # 尝试使用latin-1解码作为备用方案
                json_string = decrypted_data.decode('latin-1')
            
            print(f"调试信息 - 解密后的JSON字符串: {json_string}")
            print(f"调试信息 - JSON字符串长度: {len(json_string)}")
            
            # 检查解密数据是否为空
            if not json_string.strip():
                raise ValueError("解密后的数据为空")
            
            # 解析JSON
            try:
                payload = json.loads(json_string)
                return payload.get('unique_id')
            except json.JSONDecodeError as e:
                print(f"调试信息 - JSON解析错误: {e}")
                print(f"调试信息 - 尝试解析的字符串: '{json_string}'")
                raise ValueError(f"JSON解析失败: {e}")
        else:
            raise ValueError("无效的CryptoJS加密格式")
            
    except Exception as e:
        print(f"解密token失败: {e}")
        raise HTTPException(status_code=400, detail=f"Token解密失败: {str(e)}")

@router.post("/free_login")
async def free_login(
    body: FreeLoginRequest, 
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> JSONResponse:
    """免费令牌登录，优先验证authorization，如果存在且有效则刷新token，否则使用body中的token进行解密登录"""
    
    # 1. 优先检查authorization header
    if credentials and credentials.credentials:
        try:
            # 尝试获取当前用户信息
            # 首先验证token并获取用户
            user = await get_current_user(credentials, db)
            current_user = await get_current_active_user(user)
            if current_user:
                # 用户存在且token有效，生成新的access_token
                access_token = create_access_token(data={"sub": current_user["username"], "user_id": str(current_user["id"])})
                
                # 更新最后登录时间
                UserDAO.update_last_login(db, str(current_user["id"]))
                
                return success_response(
                    data={
                        "access_token": access_token,
                        "token_type": "bearer",
                        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600,
                        "user": current_user
                    },
                    message="用户token刷新成功"
                )
        except HTTPException as e:
            # token过期或无效，抛出异常
            raise e
        except Exception as e:
            # 其他错误，抛出异常
            raise e
    
    # 2. 如果没有authorization或authorization无效，使用body中的token
    token = body.token
    
    try:
        # 使用对称加密解密token获取唯一ID
        unique_id = decrypt_token(token)
        if not unique_id:
            return error_response(
                message="token中缺少唯一ID",
                error_code=ErrorCodes.INVALID_TOKEN,
                status_code=401
            )
        
        # 根据唯一ID查找或创建用户
        user = UserDAO.get_or_create_user_by_unique_id(db, unique_id)
        if not user:
            return error_response(
                message="获取或创建用户失败",
                error_code=ErrorCodes.INTERNAL_ERROR,
                status_code=500
            )
        
        if not user.is_active:
            return error_response(
                message="用户账户已被禁用",
                error_code=ErrorCodes.INACTIVE_USER,
                status_code=400
            )
        
        # 更新最后登录时间
        UserDAO.update_last_login(db, str(user.id))
        
        # 创建访问令牌
        access_token = create_access_token(data={"sub": user.username, "user_id": str(user.id)})
        
        user_response = UserResponse.model_validate(user)
        
        # 返回登录结果
        return success_response(
            data={
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600,
                "user": user_response.model_dump()
            },
            message="免费账户登录成功"
        )
        
    except ValueError as e:
        return error_response(
            message=str(e),
            error_code=ErrorCodes.INVALID_TOKEN,
            status_code=401
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return error_response(
            message="免费登录过程中发生错误",
            error_code=ErrorCodes.INTERNAL_ERROR,
            status_code=500,
            details={"error": str(e)}
        )
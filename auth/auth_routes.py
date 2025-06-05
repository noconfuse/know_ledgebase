from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

from models.database import get_db
from dao.user_dao import UserDAO
from .schemas import UserCreate, UserLogin, UserResponse, Token, UserUpdate
from .dependencies import get_current_user, get_current_active_user
from config import settings
from common.response import success_response, error_response, ErrorCodes
from common.exception_handler import setup_exception_handlers

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
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
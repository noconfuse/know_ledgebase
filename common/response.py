from typing import Any, Optional, Dict, Union
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import traceback
from datetime import datetime

class ApiResponse(BaseModel):
    """统一API响应格式"""
    success: bool
    message: str
    data: Optional[Any] = None
    error_code: Optional[str] = None
    timestamp: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ApiError(BaseModel):
    """API错误详情"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

def success_response(
    data: Any = None, 
    message: str = "操作成功",
    status_code: int = 200
) -> JSONResponse:
    """创建成功响应"""
    response_data = ApiResponse(
        success=True,
        message=message,
        data=data,
        timestamp=datetime.now().isoformat()
    )
    return JSONResponse(
        status_code=status_code,
        content=response_data.dict()
    )

def error_response(
    message: str,
    error_code: str = "INTERNAL_ERROR",
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None
) -> JSONResponse:
    """创建错误响应"""
    error_detail = ApiError(
        code=error_code,
        message=message,
        details=details,
        trace_id=trace_id
    )
    
    response_data = ApiResponse(
        success=False,
        message=message,
        data=None,
        error_code=error_code,
        timestamp=datetime.now().isoformat()
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data.dict()
    )

def validation_error_response(
    errors: list,
    message: str = "请求参数验证失败"
) -> JSONResponse:
    """创建验证错误响应"""
    return error_response(
        message=message,
        error_code="VALIDATION_ERROR",
        status_code=422,
        details={"validation_errors": errors}
    )

def not_found_response(
    message: str = "资源未找到",
    resource_type: str = "resource"
) -> JSONResponse:
    """创建404响应"""
    return error_response(
        message=message,
        error_code="NOT_FOUND",
        status_code=404,
        details={"resource_type": resource_type}
    )

def unauthorized_response(
    message: str = "未授权访问"
) -> JSONResponse:
    """创建401响应"""
    return error_response(
        message=message,
        error_code="UNAUTHORIZED",
        status_code=401
    )

def forbidden_response(
    message: str = "禁止访问"
) -> JSONResponse:
    """创建403响应"""
    return error_response(
        message=message,
        error_code="FORBIDDEN",
        status_code=403
    )

def bad_request_response(
    message: str = "请求参数错误",
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """创建400响应"""
    return error_response(
        message=message,
        error_code="BAD_REQUEST",
        status_code=400,
        details=details
    )

# 常用错误代码
class ErrorCodes:
    # 通用错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    BAD_REQUEST = "BAD_REQUEST"
    
    # 认证相关
    USER_ALREADY_EXISTS = "USER_ALREADY_EXISTS"
    EMAIL_ALREADY_EXISTS = "EMAIL_ALREADY_EXISTS"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    INACTIVE_USER = "INACTIVE_USER"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    
    # 业务相关
    INDEX_NOT_FOUND = "INDEX_NOT_FOUND"
    INDEX_LOAD_FAILED = "INDEX_LOAD_FAILED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    PARSING_FAILED = "PARSING_FAILED"
    USER_NOT_FOUND= "USER_NOT_FOUND"
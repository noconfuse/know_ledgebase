from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
import logging
import traceback
from typing import Union

from .response import (
    error_response, 
    validation_error_response,
    ErrorCodes
)

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """处理HTTP异常"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    # 根据状态码映射错误代码
    error_code_map = {
        400: ErrorCodes.BAD_REQUEST,
        401: ErrorCodes.UNAUTHORIZED,
        403: ErrorCodes.FORBIDDEN,
        404: ErrorCodes.NOT_FOUND,
        422: ErrorCodes.VALIDATION_ERROR,
        500: ErrorCodes.INTERNAL_ERROR
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCodes.INTERNAL_ERROR)
    
    return error_response(
        message=str(exc.detail),
        error_code=error_code,
        status_code=exc.status_code
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """处理请求验证异常"""
    logger.warning(f"Validation Error: {exc.errors()}")
    
    # 格式化验证错误
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return validation_error_response(
        errors=formatted_errors,
        message="请求参数验证失败"
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """处理通用异常"""
    logger.error(f"Unhandled Exception: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return error_response(
        message="服务器内部错误",
        error_code=ErrorCodes.INTERNAL_ERROR,
        status_code=500,
        details={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        } if logger.level <= logging.DEBUG else None
    )

async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """处理Starlette HTTP异常"""
    logger.warning(f"Starlette HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_code_map = {
        400: ErrorCodes.BAD_REQUEST,
        401: ErrorCodes.UNAUTHORIZED,
        403: ErrorCodes.FORBIDDEN,
        404: ErrorCodes.NOT_FOUND,
        422: ErrorCodes.VALIDATION_ERROR,
        500: ErrorCodes.INTERNAL_ERROR
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCodes.INTERNAL_ERROR)
    
    return error_response(
        message=str(exc.detail),
        error_code=error_code,
        status_code=exc.status_code
    )

def setup_exception_handlers(app):
    """设置异常处理器"""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
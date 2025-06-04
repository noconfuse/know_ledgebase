# 日志系统文档

## 概述

本项目实现了一个完整的日志系统，支持多种日志类型和输出方式，为文档解析和向量存储构建提供详细的日志记录和进度跟踪。

## 功能特性

### 1. 多层次日志记录
- **应用日志**: 记录应用程序的常规运行信息
- **Docling日志**: 专门记录文档转换过程的详细信息
- **进度日志**: 以JSON Lines格式记录任务进度
- **错误日志**: 集中记录所有错误信息

### 2. 灵活的输出方式
- **控制台输出**: 实时显示日志信息
- **文件输出**: 持久化存储日志到文件
- **日志轮转**: 自动管理日志文件大小和保留期限

### 3. 详细的进度跟踪
- **任务状态**: 实时跟踪任务执行状态
- **处理阶段**: 记录任务的各个处理阶段
- **性能指标**: 记录处理时间、文件大小等指标

## 配置选项

在 `config.py` 中可以配置以下日志相关选项：

```python
# 日志配置
LOG_LEVEL: str = "INFO"                    # 日志级别
LOG_FILE: str = "app.log"                  # 主日志文件名
LOG_DIR: str = "/path/to/logs"              # 日志目录
ENABLE_FILE_LOGGING: bool = True           # 启用文件日志
ENABLE_DOCLING_LOGGING: bool = True        # 启用Docling日志
LOG_ROTATION_SIZE: str = "10MB"            # 日志轮转大小
LOG_RETENTION_DAYS: int = 30               # 日志保留天数
```

## 日志文件说明

### 1. app.log
主应用日志文件，记录：
- 服务启动和关闭信息
- API请求和响应
- 任务创建和状态变更
- 一般性错误和警告

### 2. docling_processing.log
Docling文档处理专用日志，记录：
- 文档转换过程
- OCR处理详情
- 表格和图像提取
- 转换性能指标

### 3. progress.jsonl
进度跟踪日志（JSON Lines格式），每行包含：
```json
{
  "timestamp": "2025-06-04T01:25:55.604707",
  "task_id": "4299d252-11cd-48d3-8dcf-6c08263fb8b9",
  "progress": 50,
  "stage": "文档解析",
  "message": "正在提取表格",
  "details": {"tables_found": 3}
}
```

### 4. errors.log
错误日志文件，集中记录：
- 异常堆栈信息
- 系统错误
- 文件处理错误
- 配置错误

## API接口

### 1. 获取任务日志
```http
GET /parse/logs/{task_id}?limit=50
```
返回指定任务的处理日志。

### 2. 下载日志文件
```http
GET /logs/download?log_type=main&date=2025-06-04
```
下载指定类型和日期的日志文件。

### 3. 列出日志文件
```http
GET /logs/list
```
列出所有可用的日志文件及其信息。

## 使用示例

### 1. 基本日志记录
```python
from utils.logging_config import setup_logging, get_logger

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# 记录日志
logger.info("应用启动")
logger.warning("配置文件缺少某些选项")
logger.error("处理文件时出错")
```

### 2. 进度日志记录
```python
from utils.logging_config import log_progress

# 记录进度
log_progress("task_123", 50, "文档解析", {
    "current_page": 5,
    "total_pages": 10
})
```

### 3. 任务处理日志
```python
# 在ParseTask中记录处理日志
task.add_processing_log("info", "开始解析文档", {
    "file_size": 1024000,
    "file_type": "pdf"
})
```

## 日志级别说明

- **DEBUG**: 详细的调试信息，仅在开发时使用
- **INFO**: 一般信息，记录正常的程序流程
- **WARNING**: 警告信息，程序可以继续运行但需要注意
- **ERROR**: 错误信息，程序遇到错误但可以恢复
- **CRITICAL**: 严重错误，程序可能无法继续运行

## 性能考虑

1. **异步日志**: 使用异步方式写入日志文件，避免阻塞主线程
2. **日志轮转**: 自动管理日志文件大小，防止磁盘空间耗尽
3. **缓冲写入**: 批量写入日志，提高I/O效率
4. **级别过滤**: 根据配置的日志级别过滤不必要的日志

## 故障排除

### 1. 日志文件未创建
- 检查日志目录权限
- 确认 `ENABLE_FILE_LOGGING` 配置为 `True`
- 查看控制台是否有权限错误信息

### 2. 日志内容缺失
- 检查日志级别配置
- 确认日志处理器正确初始化
- 查看是否有异常导致日志记录中断

### 3. 性能问题
- 调整日志级别，减少不必要的日志
- 增加日志轮转频率
- 检查磁盘I/O性能

## 最佳实践

1. **合理设置日志级别**: 生产环境使用INFO或WARNING级别
2. **定期清理日志**: 设置合适的日志保留期限
3. **监控日志大小**: 避免日志文件过大影响性能
4. **结构化日志**: 使用一致的日志格式，便于分析
5. **敏感信息保护**: 避免在日志中记录密码、密钥等敏感信息

## 扩展功能

未来可以考虑添加的功能：
- 日志聚合和分析
- 实时日志监控
- 日志告警机制
- 分布式日志收集
- 日志可视化界面
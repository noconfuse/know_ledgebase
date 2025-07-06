# MinerU集成文档

## 概述

本项目已成功集成MinerU文档解析器，提供了与现有Docling解析器并行的PDF解析能力。MinerU专门针对PDF文档进行了优化，能够更好地处理复杂的PDF布局和内容提取。

## 功能特性

### MinerU解析器特点
- **专业PDF解析**: 专门针对PDF文档优化
- **高质量内容提取**: 支持文本、图片、表格的精确提取
- **结构化输出**: 提供Markdown格式和结构化JSON数据
- **异步处理**: 支持异步任务处理，提高并发性能
- **进度跟踪**: 实时任务进度和状态监控

### 支持的功能
- PDF文档解析
- 文本内容提取
- 图片内容识别
- 表格结构解析
- 文档结构分析（标题层级）
- Markdown格式输出
- 元数据提取

## API接口

### 1. 文件上传解析

**端点**: `POST /parse/upload`

**参数**:
- `file`: 上传的PDF文件
- `parser_type`: 解析器类型，可选值：`"docling"` 或 `"mineru"`
- `config`: 解析配置（可选）

**示例**:
```bash
curl -X POST "http://localhost:8001/parse/upload" \
  -F "file=@document.pdf" \
  -F "parser_type=mineru" \
  -F "config={\"save_to_file\": true, \"max_workers\": 1}"
```

### 2. 文件路径解析

**端点**: `POST /parse/file`

**请求体参数**:
- `file_path`: 文件路径（必需）
- `config`: 解析配置对象（可选）
  - `parser_type`: 解析器类型（可选）
  - `ocr_enabled`: 是否启用OCR（可选）
  - `max_workers`: MinerU并发数（可选）
  - 其他配置参数...

**示例**:
```bash
curl -X POST "http://localhost:8001/parse/file" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "config": {
      "parser_type": "mineru",
      "max_workers": 1,
      "ocr_enabled": true
    }
  }'
```

### 2.1 目录解析

**端点**: `POST /parse/directory`

**请求体参数**:
- `directory_path`: 目录路径（必需）
- `config`: 解析配置对象（可选）

**示例**:
```bash
curl -X POST "http://localhost:8001/parse/directory" \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/path/to/documents/",
    "config": {
      "parser_type": "docling",
      "ocr_enabled": false
    }
  }'
```

### 3. 任务状态查询

**端点**: `GET /parse/status/{task_id}`

**响应示例**:
```json
{
  "task_id": "abc123",
  "status": "processing",
  "progress": 65,
  "current_stage": "MinerU解析",
  "parser_type": "mineru",
  "created_at": 1638360000.0,
  "started_at": 1638360001.0
}
```

### 4. 任务结果获取

**端点**: `GET /parse/result/{task_id}`

**响应示例**:
```json
{
  "document_id": "doc123",
  "title": "文档标题",
  "content": "# 文档标题\n\n文档内容...",
  "parser_type": "mineru",
  "content_length": 1500,
  "has_tables": true,
  "has_images": false,
  "statistics": {
    "text_blocks": 25,
    "image_blocks": 0,
    "table_blocks": 3,
    "total_blocks": 28
  },
  "metadata": {
    "file_size": 2048576,
    "output_directory": "/path/to/output",
    "output_files": {
      "markdown": "document.md",
      "json": "document.json"
    }
  },
  "structure": {
    "headings": [
      {"level": 1, "text": "第一章", "page": 1},
      {"level": 2, "text": "1.1 概述", "page": 1}
    ],
    "outline": ["第一章", "1.1 概述"]
  }
}
```

## 配置选项

### ParseConfig参数

```python
class ParseConfig(BaseModel):
    parser_type: Optional[str] = None  # "docling" 或 "mineru"
    ocr_enabled: Optional[bool] = False  # 是否启用OCR
    ocr_languages: Optional[List[str]] = []  # OCR语言
    extract_tables: Optional[bool] = True  # 是否提取表格
    extract_images: Optional[bool] = True  # 是否提取图片信息
    max_workers: Optional[int] = None  # MinerU解析器的并发数量（仅对MinerU有效）
```

### 环境配置

在 `config.py` 中可以配置以下参数：

```python
# MinerU配置
DEFAULT_PARSER_TYPE = "docling"  # 默认解析器
OUTPUT_DIR = "/path/to/mineru/output"  # MinerU输出目录
MINERU_MAX_WORKERS = 1  # MinerU最大并发数
```

## 使用示例

### Python客户端示例

```python
import requests
import time

# 1. 直接文件路径解析
response = requests.post(
    'http://localhost:8001/parse/file',
    json={
        'file_path': '/path/to/document.pdf',
        'config': {
            'parser_type': 'mineru',
            'max_workers': 1,
            'ocr_enabled': True
        }
    }
)

task_id = response.json()['task_id']
print(f"任务ID: {task_id}")

# 2. 轮询任务状态
while True:
    status_response = requests.get(f'http://localhost:8001/parse/status/{task_id}')
    status = status_response.json()
    
    print(f"状态: {status['status']}, 进度: {status['progress']}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(2)

# 3. 获取解析结果
if status['status'] == 'completed':
    result_response = requests.get(f'http://localhost:8001/parse/result/{task_id}')
    result = result_response.json()
    
    print(f"解析完成！")
    print(f"文档标题: {result['title']}")
    print(f"内容长度: {result['content_length']}")
    print(f"包含表格: {result['has_tables']}")
    print(f"包含图片: {result['has_images']}")
else:
    print(f"解析失败: {status.get('error')}")
```

### JavaScript客户端示例

```javascript
// 文件路径解析
async function parseDocument(filePath) {
    const response = await fetch('/parse/file', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_path: filePath,
            config: {
                parser_type: 'mineru',
                max_workers: 1,
                ocr_enabled: true
            }
        })
    });
    
    const result = await response.json();
    return result.task_id;
}

// 轮询任务状态
async function waitForCompletion(taskId) {
    while (true) {
        const response = await fetch(`/parse/status/${taskId}`);
        const status = await response.json();
        
        console.log(`状态: ${status.status}, 进度: ${status.progress}%`);
        
        if (status.status === 'completed') {
            return await getResult(taskId);
        } else if (status.status === 'failed') {
            throw new Error(status.error);
        }
        
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}

// 获取解析结果
async function getResult(taskId) {
    const response = await fetch(`/parse/result/${taskId}`);
    return await response.json();
}
```

## 解析器对比

| 特性 | Docling | MinerU |
|------|---------|--------|
| 支持格式 | PDF, DOCX, DOC, TXT, MD, HTML, PPTX, XLSX | PDF |
| PDF解析质量 | 良好 | 优秀 |
| 表格提取 | 支持 | 优秀 |
| 图片提取 | 支持 | 优秀 |
| 文档结构 | 基本 | 详细 |
| 处理速度 | 快速 | 中等 |
| 内存占用 | 低 | 中等 |
| OCR支持 | 内置 | 依赖外部 |

## 最佳实践

### 1. 解析器选择
- **PDF文档**: 推荐使用MinerU，特别是包含复杂表格和图片的PDF
- **其他格式**: 使用Docling
- **批量处理**: 根据文档类型混合使用

### 2. 性能优化
- 对于大文件，设置合适的`max_workers`参数
- 使用异步处理避免阻塞
- 定期清理完成的任务

### 3. 错误处理
- 实现任务状态轮询机制
- 处理解析失败的情况
- 设置合理的超时时间

### 4. 存储管理
- 定期清理输出目录
- 监控磁盘空间使用
- 备份重要的解析结果

## 故障排除

### 常见问题

1. **MinerU解析失败**
   - 检查PDF文件是否损坏
   - 确认文件大小在限制范围内
   - 查看错误日志获取详细信息

2. **任务卡住不动**
   - 检查系统资源使用情况
   - 重启服务
   - 清理临时文件

3. **内存不足**
   - 减少`max_workers`参数
   - 增加系统内存
   - 分批处理大文件

### 日志查看

```bash
# 查看解析日志
tail -f /path/to/logs/document_parsing.log

# 查看错误日志
tail -f /path/to/logs/errors.log

# 查看进度日志
tail -f /path/to/logs/progress.jsonl
```

## 更新日志

### v1.0.0 (2024-06-04)
- ✅ 集成MinerU解析器
- ✅ 支持PDF专业解析
- ✅ 异步任务处理
- ✅ 进度跟踪和状态监控
- ✅ API接口扩展
- ✅ 配置参数支持
- ✅ 文档结构提取
- ✅ 统计信息收集

## 技术支持

如有问题，请查看：
1. 本文档的故障排除部分
2. 系统日志文件
3. GitHub Issues

---

*最后更新: 2024-06-04*
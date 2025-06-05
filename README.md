# LlamaIndex RAG 知识库系统

基于 LlamaIndex 框架的智能文档解析和检索增强生成(RAG)系统，采用前后端分离架构，提供两个独立的 FastAPI 服务。

## 🚀 项目特性

### 后台服务 (端口 8001) - Backend Service
- ✅ 基于 Docling 的高质量文档解析
- ✅ 支持多种文档格式：PDF、DOCX、DOC、TXT、MD、HTML、PPTX、XLSX
- ✅ OCR 支持，GPU 加速处理图片和扫描件
- ✅ 异步任务处理，支持进度查询
- ✅ 可配置解析参数
- ✅ 可选本地文件保存
- ✅ 向量数据库构建功能
- ✅ Chunk 级别关键词提取
- ✅ 文档级别摘要生成
- ✅ 文档级别问答对生成
- ✅ 统一API响应格式
- ✅ 全局异常处理

### 前台服务 (端口 8002) - Frontend Service
- ✅ 混合检索（向量检索 + BM25 + 重排序）
- ✅ 检索召回测试接口
- ✅ 第三方 LLM API 支持，可灵活切换
- ✅ 流式对话接口
- ✅ 多轮对话支持
- ✅ 会话管理
- ✅ 用户认证和授权
- ✅ 统一API响应格式
- ✅ 全局异常处理
- ✅ LlamaIndex 最佳实践

## 📋 系统要求

- Python 3.8+
- CUDA 支持的 GPU (推荐)
- 8GB+ RAM
- 10GB+ 磁盘空间

## 🛠️ 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd know_ledgebase
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
创建 `.env` 文件：
```bash
# 模型路径配置
MODEL_BASE_DIR=/path/to/your/models
KNOWLEDGE_BASE_DIR=/path/to/your/knowledge_base

# LLM API 配置 (可选)
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=your_api_key
```

### 5. 下载模型
确保以下模型已下载到 `MODEL_BASE_DIR`：
- `gte-base-zh` (嵌入模型)
- `bge-reranker-large` (重排序模型)
- `internlm2_5-1_8b-chat` (LLM模型，如使用本地模型)

## 🚀 快速启动

### 方式一：使用启动脚本（推荐）
```bash
python start_services.py
```

### 方式二：分别启动服务

**启动后台服务（文档解析）：**
```bash
uvicorn apps.backend_service:app --host 0.0.0.0 --port 8001 --reload
```

**启动前台服务（RAG检索）：**
```bash
uvicorn apps.frontend_service:app --host 0.0.0.0 --port 8002 --reload
```

## 📖 API 使用指南

### 后台服务 API (端口 8001)

#### 1. 上传文件解析
```bash
curl -X POST "http://localhost:8001/parse/upload" \
  -F "file=@document.pdf" \
  -F "save_to_file=true" \
  -F 'config={"ocr_enabled": true}'
```

#### 2. 解析本地文件
```bash
curl -X POST "http://localhost:8001/parse/file" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "config": {
      "ocr_enabled": true,
      "save_to_file": true
    }
  }'
```

#### 3. 查询解析状态
```bash
curl "http://localhost:8001/parse/status/{task_id}"
```

#### 4. 构建向量数据库
```bash
curl -X POST "http://localhost:8001/vector-store/build" \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/path/to/documents",
    "config": {
      "extract_keywords": true,
      "extract_summary": true,
      "generate_qa": true,
      "chunk_size": 512
    }
  }'
```

### 前台服务 API (端口 8002)

#### 1. 加载向量索引
```bash
curl -X POST "http://localhost:8002/index/load?index_id={index_id}"
```

#### 2. 混合检索
```bash
curl -X POST "http://localhost:8002/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "index_id": "your_index_id",
    "query": "你的问题",
    "top_k": 5
  }'
```

#### 3. 创建对话会话
```bash
curl -X POST "http://localhost:8002/chat/session" \
  -H "Content-Type: application/json" \
  -d '{
    "index_id": "your_index_id"
  }'
```

#### 4. 非流式对话
```bash
curl -X POST "http://localhost:8002/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "message": "你的问题"
  }'
```

#### 5. 流式对话
```bash
curl -X POST "http://localhost:8002/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "message": "你的问题"
  }'
```

#### 6. 检索召回测试
```bash
curl -X POST "http://localhost:8002/test/recall" \
  -H "Content-Type: application/json" \
  -d '{
    "index_id": "your_index_id",
    "test_queries": [
      {
        "query": "测试问题1",
        "expected_docs": ["doc1.pdf", "doc2.pdf"],
        "top_k": 5
      }
    ]
  }'
```

## 🏗️ 项目架构

```
know_ledgebase/
├── apps/                    # FastAPI 应用服务
│   ├── backend_service.py   # 后台服务（文档解析、向量构建）
│   └── frontend_service.py  # 前台服务（RAG检索、对话）
├── auth/                    # 认证模块
│   ├── auth_routes.py       # 认证路由
│   ├── dependencies.py      # 认证依赖
│   └── schemas.py          # 认证数据模型
├── common/                  # 公共模块
│   ├── response.py         # 统一响应格式
│   └── exception_handler.py # 全局异常处理
├── models/                  # 数据模型
│   ├── user_models.py      # 用户模型
│   ├── user_dao.py         # 用户数据访问
│   ├── chat_models.py      # 聊天模型
│   └── chat_dao.py         # 聊天数据访问
├── services/                # 核心服务模块
│   ├── document_parser.py   # 文档解析器
│   ├── vector_store_builder.py # 向量数据库构建器
│   └── rag_service.py       # RAG 检索服务
├── utils/                   # 工具模块
│   └── logging_config.py    # 日志配置
├── config.py               # 配置文件
├── requirements.txt        # 依赖列表
├── start_services.py       # 启动脚本
└── README.md              # 项目文档
```

## ⚙️ 配置说明

### 主要配置项

- `MODEL_BASE_DIR`: 模型文件目录
- `KNOWLEDGE_BASE_DIR`: 知识库数据目录
- `USE_GPU`: 是否使用 GPU
- `OCR_ENABLED`: 是否启用 OCR
- `CHUNK_SIZE`: 文本分块大小
- `RETRIEVAL_TOP_K`: 检索返回数量
- `LLM_API_BASE`: LLM API 地址
- `LLM_API_KEY`: LLM API 密钥

### 支持的文档格式

- PDF (.pdf)
- Word (.docx, .doc)
- 文本 (.txt, .md)
- 网页 (.html)
- 演示文稿 (.pptx)
- 电子表格 (.xlsx)

## 🔧 开发指南

### 添加新的文档解析器

1. 在 `services/document_parser.py` 中扩展 `_parse_file_sync` 方法
2. 添加新的文件格式支持
3. 更新配置中的 `SUPPORTED_FORMATS`

### 添加新的检索器

1. 在 `services/rag_service.py` 中扩展 `hybrid_retrieve` 方法
2. 实现新的检索算法
3. 更新融合检索逻辑

### 自定义 LLM

1. 在 `services/rag_service.py` 中修改 `_setup_llm` 方法
2. 添加新的 LLM 提供商支持
3. 更新配置参数

## 📊 性能优化

### GPU 加速
- 确保安装了 CUDA 版本的 PyTorch
- 设置 `USE_GPU=True`
- 配置合适的 `GPU_DEVICE`

### 内存优化
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 限制 `MAX_CONCURRENT_TASKS`
- 定期清理过期任务

### 检索优化
- 调整 `RETRIEVAL_TOP_K` 和 `RERANK_TOP_K`
- 设置合适的 `SIMILARITY_THRESHOLD`
- 使用合适的嵌入模型

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的内存和显存
   - 验证模型文件完整性

2. **OCR 不工作**
   - 检查 GPU 驱动和 CUDA 版本
   - 确保 OCR 模型已下载
   - 检查图片格式支持

3. **检索结果不准确**
   - 调整相似度阈值
   - 检查文档分块质量
   - 优化查询表述

4. **服务启动失败**
   - 检查端口是否被占用
   - 验证依赖安装完整性
   - 查看详细错误日志

### 日志查看

```bash
# 查看服务日志
tail -f app.log

# 查看特定服务日志
grep "Backend Service" app.log
grep "Frontend Service" app.log
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请提交 Issue 或联系开发团队。
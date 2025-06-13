# 向量化API配置文档

本文档描述了向量化流程API的配置参数，包括新增的智能元数据提取器配置。

## API端点

### 构建向量存储

**POST** `/vector-store/build`

#### 请求参数

```json
{
  "directory_path": "string",  // 必需：文档目录路径
  "config": {                   // 可选：配置参数
    // 基础配置
    "chunk_size": 512,
    "chunk_overlap": 50,
    
    // 智能元数据提取器配置
    "extract_mode": "enhanced",
    "min_chunk_size_for_summary": 500,
    "min_chunk_size_for_qa": 300,
    "max_keywords": 5,
    "num_questions": 3,
    
    // 索引描述配置
    "index_description": "可选的索引描述信息"
  }
}
```

#### 配置参数详解

##### 基础配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `chunk_size` | integer | `512` | 文本块大小（字符数） |
| `chunk_overlap` | integer | `50` | 文本块重叠大小 |

##### 智能元数据提取器配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `extract_mode` | string | `"enhanced"` | 提取模式：`"basic"` 或 `"enhanced"` |
| `min_chunk_size_for_summary` | integer | `500` | 提取摘要的最小chunk大小（仅enhanced模式） |
| `min_chunk_size_for_qa` | integer | `300` | 提取问答对的最小chunk大小 |
| `max_keywords` | integer | `5` | 最大关键词数量 |
| `num_questions` | integer | `3` | 问答对数量 |

##### 索引配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `index_description` | string | `null` | 索引描述信息 |

#### 响应

```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "message": "Vector store build task created successfully"
}
```

### 获取任务状态

**GET** `/vector-store/status/{task_id}`

#### 响应

```json
{
  "task_id": "uuid-string",
  "status": "running|completed|failed",
  "progress": 75,
  "current_stage": "processing documents",
  "result": {
    "index_id": "uuid-string",
    "document_count": 10,
    "node_count": 150,
    "config": {...}
  },
  "error": null,
  "created_at": 1234567890.0,
  "started_at": 1234567890.0,
  "completed_at": 1234567890.0
}
```

## 配置模式推荐

### 1. 高性能模式（快速处理）

适用于需要快速处理大量文档的场景。

```json
{
  "extract_mode": "basic",
  "min_chunk_size_for_summary": 1000,
  "min_chunk_size_for_qa": 800,
  "max_keywords": 3,
  "num_questions": 2,
  "chunk_size": 256,
  "chunk_overlap": 25
}
```

**特点：**
- 减少LLM调用次数
- 较小的chunk大小，处理更快
- 基础元数据提取

### 2. 平衡模式（推荐）

适用于大多数应用场景，平衡处理速度和元数据质量。

```json
{
  "extract_mode": "enhanced",
  "min_chunk_size_for_summary": 500,
  "min_chunk_size_for_qa": 300,
  "max_keywords": 5,
  "num_questions": 3,
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

**特点：**
- 默认配置
- 适中的处理速度
- 良好的元数据质量

### 3. 高质量模式（详细元数据）

适用于对元数据质量要求较高的场景。

```json
{
  "extract_mode": "enhanced",
  "min_chunk_size_for_summary": 200,
  "min_chunk_size_for_qa": 150,
  "max_keywords": 8,
  "num_questions": 5,
  "chunk_size": 1024,
  "chunk_overlap": 100
}
```

**特点：**
- 更低的提取阈值
- 更多的关键词和问答对
- 更大的chunk包含更多上下文

### 4. 成本优化模式（最少LLM调用）

适用于成本敏感的场景。

```json
{
  "extract_mode": "basic",
  "min_chunk_size_for_summary": 2000,
  "min_chunk_size_for_qa": 1500,
  "max_keywords": 3,
  "num_questions": 1,
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

**特点：**
- 很高的提取阈值
- 最少的LLM调用
- 基础元数据提取

### 5. 法律文档专用模式

适用于法律文档处理。

```json
{
  "extract_mode": "enhanced",
  "min_chunk_size_for_summary": 400,
  "min_chunk_size_for_qa": 250,
  "max_keywords": 6,
  "num_questions": 4,
  "chunk_size": 768,
  "chunk_overlap": 75,
  "index_description": "法律文档知识库，包含法律条文、司法解释和案例分析"
}
```

**特点：**
- 适合法律条文的chunk大小
- 更多的关键词和问答对
- 包含索引描述

## 智能阈值控制

新的智能元数据提取器根据chunk大小智能决定提取哪些元数据：

- **文档级元数据**：每个文档只提取一次，自动继承到所有chunk
- **基础chunk元数据**：所有chunk都提取（标题、关键词）
- **摘要**：仅当chunk大小 ≥ `min_chunk_size_for_summary` 时提取
- **问答对**：仅当chunk大小 ≥ `min_chunk_size_for_qa` 时提取

这种设计避免了对短chunk进行低质量的元数据提取，同时确保长chunk获得完整的元数据。

## 性能优化建议

1. **调整阈值参数**：根据文档特点调整 `min_chunk_size_for_summary` 和 `min_chunk_size_for_qa`
2. **选择合适的模式**：根据应用场景选择 `extract_mode`
3. **控制元数据数量**：适当设置 `max_keywords` 和 `num_questions`
4. **优化chunk大小**：根据文档类型调整 `chunk_size` 和 `chunk_overlap`

## 向后兼容性

新的API完全向后兼容，原有的配置参数仍然有效。如果不提供新的智能元数据提取器参数，系统将使用默认值。

## 错误处理

- `400 Bad Request`：配置参数无效
- `404 Not Found`：目录不存在
- `500 Internal Server Error`：服务器内部错误

## 示例代码

参考 `examples/api_config_examples.py` 文件获取完整的使用示例。
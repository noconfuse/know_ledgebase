# DocumentPreservingTransformer 使用说明

## 概述

`DocumentPreservingTransformer` 是一个自定义的节点解析器，它在对文档进行切分的同时保留完整的原始文档节点。这解决了传统纯切分策略在处理文档级查询时可能出现的信息碎片化问题。

## 功能特点

1. **双层节点策略**：为每个文档创建一个完整的原始文档节点和多个切分节点
2. **元数据标记**：通过 `node_type` 和 `is_complete_document` 元数据区分节点类型
3. **兼容性**：与现有的 LlamaIndex 节点解析器完全兼容
4. **灵活配置**：支持各种文件类型的不同解析策略

## 节点类型

### 原始文档节点
- `node_type`: "original_document"
- `is_complete_document`: True
- 包含完整的文档内容
- 适用于文档级查询和概览

### 切分节点
- `node_type`: "chunk"
- `is_complete_document`: False
- 包含文档的片段内容
- 适用于精确的语义检索

## 实现原理

`DocumentPreservingTransformer` 继承自 `NodeParser`，在 `_parse_nodes` 方法中：

1. 为每个输入文档创建一个原始文档节点（保留完整内容）
2. 使用指定的基础解析器对文档进行切分
3. 为所有节点添加适当的元数据标记
4. 返回包含原始节点和切分节点的完整列表

## 使用示例

```python
from services.vector_store_builder import DocumentPreservingTransformer
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 创建文档
doc = Document(
    text="这是一个测试文档...",
    metadata={'file_name': 'test.txt'}
)

# 创建转换器
transformer = DocumentPreservingTransformer(
    SentenceSplitter(chunk_size=150, chunk_overlap=30)
)

# 处理文档
nodes = transformer._parse_nodes([doc])

# 分析结果
original_nodes = [n for n in nodes if n.metadata.get('node_type') == 'original_document']
chunk_nodes = [n for n in nodes if n.metadata.get('node_type') == 'chunk']
```

## 在 VectorStoreBuilder 中的集成

该转换器已集成到 `VectorStoreBuilder` 的 `_get_node_parser_for_file_type` 方法中，会自动应用于所有文件类型：

- Markdown 文件：MarkdownNodeParser + SentenceSplitter
- HTML 文件：HTMLNodeParser + SentenceSplitter  
- JSON 文件：DoclingNodeParser
- 其他文本文件：SentenceSplitter

## 优势

1. **解决信息碎片化**：保留完整文档避免重要信息丢失
2. **提升检索质量**：支持文档级和片段级两种检索策略
3. **改善用户体验**：可以提供完整的文档上下文
4. **保持向后兼容**：不影响现有的切分节点功能

## 注意事项

1. 会增加节点总数（每个文档额外增加1个原始节点）
2. 会增加存储空间需求
3. 在检索时需要根据查询类型选择合适的节点类型

## 测试

可以运行 `test_document_preserving.py` 来验证功能：

```bash
python3 test_document_preserving.py
```

该测试会创建一个长文档，验证原始节点和切分节点的正确创建。
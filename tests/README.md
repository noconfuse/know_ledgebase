# RAG召回准确率测试套件

本测试套件用于评估RAG（检索增强生成）系统在法律文档上的召回准确率，特别针对消防法文档进行优化。

## 文件说明

### 核心文件

1. **test_data_generator.py** - 测试数据生成器
   - 生成多样化的测试查询集合
   - 包含24个精心设计的测试用例
   - 覆盖定义、职责、程序、处罚等多个类别

2. **enhanced_recall_test.py** - 增强版召回测试脚本
   - 全面的召回准确率评估
   - 多维度指标分析（关键词召回、条文召回、语义相关性等）
   - 详细的分类统计和报告生成

3. **run_quick_test.py** - 快速测试脚本
   - 快速验证RAG服务基本功能
   - 适合在运行完整测试前进行预检

4. **test_cases.json** - 测试用例数据文件
   - 由test_data_generator.py生成
   - 包含所有测试查询及其期望结果

### 辅助文件

- **rag_recall_test.py** - 基础召回测试脚本（较简单版本）
- **quick_recall_test.py** - 简化的快速测试脚本

## 测试用例分类

### 按类别分布
- **基础定义** (3个): 消防设施、消防产品、人员密集场所等定义
- **政府职责** (1个): 政府在消防工作中的职责
- **部门职责** (1个): 应急管理部门的监督管理职责
- **公民义务** (1个): 公民在消防工作中的义务
- **企业职责** (1个): 物业服务企业的消防职责
- **审查程序** (1个): 建设工程消防设计审查流程
- **验收程序** (1个): 消防验收程序
- **检查程序** (1个): 公众聚集场所消防安全检查
- **建设违法** (1个): 未经消防设计审查擅自施工的处罚
- **违法行为** (1个): 谎报火警的法律后果
- **火灾责任** (1个): 过失引起火灾的法律责任
- **危险品场所** (1个): 易燃易爆危险品场所设置要求
- **群众活动** (1个): 大型群众性活动消防安全要求
- **明火管理** (1个): 禁止吸烟和使用明火的场所规定
- **救援队伍** (1个): 国家综合性消防救援队职责
- **消防队伍** (1个): 专职消防队和志愿消防队建立要求
- **质量监督** (1个): 消防产品质量监督体系
- **设施管理** (1个): 建筑消防设施全生命周期管理
- **宣传教育** (1个): 消防宣传教育责任主体和实施方式
- **适用范围** (1个): 森林和草原消防工作的法律适用
- **特殊管辖** (1个): 军事设施消防工作管辖
- **法律效力** (1个): 消防法施行时间

### 按难度分布
- **简单** (4个): 基础概念和定义
- **中等** (11个): 职责义务和一般规定
- **困难** (9个): 复杂程序和综合分析

### 按查询类型分布
- **定义** (3个): 概念解释类查询
- **职责** (4个): 职责义务类查询
- **程序** (3个): 流程程序类查询
- **处罚** (3个): 法律责任类查询
- **要求** (3个): 规范要求类查询
- **体系** (2个): 系统性分析类查询
- **义务** (1个): 义务责任类查询
- **管理** (1个): 管理制度类查询
- **规定** (1个): 具体规定类查询
- **管辖** (1个): 管辖权限类查询
- **适用性** (1个): 法律适用类查询
- **时效** (1个): 时间效力类查询

## 使用方法

### 1. 环境准备

确保RAG服务正在运行：
```bash
# 检查服务状态
curl http://localhost:8001/health
```

### 2. 快速测试

首先运行快速测试验证服务基本功能：
```bash
cd /home/ubuntu/workspace/know_ledgebase/tests
python3 run_quick_test.py
```

### 3. 生成测试数据

生成测试用例集合：
```bash
python3 test_data_generator.py
```

### 4. 运行完整测试

运行增强版召回准确率测试：
```bash
python3 enhanced_recall_test.py
```

### 5. 查看结果

测试完成后会生成以下文件：
- `test_results_YYYYMMDD_HHMMSS.json` - 详细测试结果
- `test_summary_YYYYMMDD_HHMMSS.json` - 摘要报告

## 评估指标

### 主要指标

1. **关键词召回率** - 检索结果中包含期望关键词的比例
2. **关键词精确率** - 找到的关键词与期望关键词的匹配度
3. **条文召回率** - 检索结果中包含相关法条的比例
4. **条文精确率** - 找到的法条与期望法条的匹配度
5. **语义相关性** - 检索结果与查询的语义相似度
6. **响应时间** - 检索请求的平均响应时间
7. **成功率** - 成功完成检索的请求比例

### 分析维度

- **总体性能** - 所有测试用例的平均表现
- **类别分析** - 按查询类别分组的性能分析
- **难度分析** - 按查询难度分组的性能分析
- **最佳/最差案例** - 表现最好和最差的查询案例

## 配置说明

### 默认配置

- **RAG服务地址**: `http://localhost:8001`
- **索引ID**: `1b70e012-79b7-4b20-8f70-9e94646e3aad`
- **检索文档数**: 10个（top_k=10）
- **请求超时**: 30秒

### 自定义配置

可以通过修改脚本中的参数来调整配置：

```python
# 在enhanced_recall_test.py中
tester = EnhancedRAGTester(
    base_url="http://your-rag-service:8001",
    index_id="your-index-id"
)

# 运行测试时指定top_k
report = tester.run_comprehensive_test(top_k=15)
```

## 结果解读

### 性能基准

- **关键词召回率 > 0.7**: 良好
- **条文召回率 > 0.6**: 良好
- **语义相关性 > 0.5**: 良好
- **响应时间 < 2秒**: 良好
- **成功率 > 0.95**: 良好

### 优化建议

1. **关键词召回率低**: 检查分词和索引策略
2. **条文召回率低**: 优化文档分块和元数据提取
3. **语义相关性低**: 调整embedding模型或相似度计算
4. **响应时间长**: 优化索引结构或增加计算资源
5. **成功率低**: 检查服务稳定性和错误处理

## 扩展测试

### 添加新测试用例

在`test_data_generator.py`中添加新的测试用例：

```python
def _generate_custom_queries(self):
    """生成自定义查询"""
    cases = [
        TestCase(
            id="custom_001",
            query="你的查询问题",
            expected_keywords=["关键词1", "关键词2"],
            expected_articles=["第XX条"],
            category="自定义类别",
            difficulty="medium",
            query_type="自定义类型",
            description="测试描述"
        )
    ]
    self.test_cases.extend(cases)
```

### 自定义评估指标

在`enhanced_recall_test.py`中添加新的评估方法：

```python
def calculate_custom_metric(self, query: str, retrieved_docs: List[Dict]) -> float:
    """计算自定义指标"""
    # 实现你的评估逻辑
    return score
```

## 故障排除

### 常见问题

1. **连接失败**: 检查RAG服务是否启动，端口是否正确
2. **索引不存在**: 确认索引ID是否正确，索引是否已构建
3. **超时错误**: 增加timeout参数或检查服务性能
4. **权限错误**: 确认文件读写权限

### 调试模式

在脚本中添加调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 许可证

本测试套件遵循项目的开源许可证。

# 测试文档

本目录包含了LlamaIndex RAG知识库系统的测试用例和验证脚本。

## 文件说明

### 1. test_api_cases.py
**接口测试用例**

这是一个全面的API接口测试工具，用于测试系统的所有REST API端点。

**功能特性：**
- 服务健康状态检查
- 文档解析API测试（文件路径和文件上传）
- 向量数据库构建API测试
- RAG检索功能测试
- 聊天对话功能测试
- 召回性能测试
- 配置API测试

**使用方法：**
```bash
# 确保服务已启动
python start_services.py

# 在另一个终端运行测试
cd /home/ubuntu/workspace/know_ledgebase
python tests/test_api_cases.py
```

**配置参数：**
- `test_file_path`: 测试文件路径
- `test_dir_path`: 测试目录路径
- `doc_service_url`: 文档服务URL（默认：http://localhost:8000）
- `rag_service_url`: RAG服务URL（默认：http://localhost:8001）

### 2. ../test_workflow.py
**完整功能验证脚本**

这是一个端到端的工作流程测试脚本，模拟真实的使用场景。

**测试流程：**
1. **文档创建和解析**
   - 自动创建AI相关的测试文档
   - 测试文档解析功能
   - 验证解析结果

2. **向量数据库构建**
   - 基于解析后的文档构建向量数据库
   - 包含关键词提取、摘要生成、问答对生成
   - 监控构建进度

3. **检索功能测试**
   - 加载向量索引
   - 执行多个测试查询
   - 评估检索质量和响应时间

4. **对话功能测试**
   - 创建聊天会话
   - 进行多轮对话测试
   - 验证上下文理解能力

**使用方法：**
```bash
# 启动服务
python start_services.py

# 运行完整工作流程测试
python test_workflow.py
```

**输出结果：**
- 控制台实时日志
- 详细的JSON测试报告
- 性能统计信息

### 3. ../quick_test.py
**快速功能测试脚本**

这是一个轻量级的快速验证脚本，用于基本功能检查。

**测试内容：**
- 服务状态检查
- 基本文档解析
- 简单向量数据库构建
- 基础检索功能
- 简单对话测试

**使用方法：**
```bash
# 启动服务
python start_services.py

# 快速测试
python quick_test.py
```

**适用场景：**
- 开发环境快速验证
- CI/CD流水线集成
- 部署后的基本功能检查

## 测试环境准备

### 1. 安装依赖
```bash
cd /home/ubuntu/workspace/know_ledgebase
pip install -r requirements.txt
```

### 2. 启动服务
```bash
# 启动所有服务
python start_services.py

# 或者分别启动
python apps/document_service.py &
python apps/rag_service_app.py &
```

### 3. 验证服务状态
```bash
# 检查文档服务
curl http://localhost:8000/health

# 检查RAG服务
curl http://localhost:8001/health
```

## 测试数据准备

### 自动生成测试数据
所有测试脚本都会自动创建所需的测试数据，无需手动准备。

### 使用自定义测试数据
如果要使用自定义测试数据，请：

1. 将文档放在 `test_data/` 目录下
2. 支持的格式：`.txt`, `.pdf`, `.docx`, `.md`
3. 修改测试脚本中的文件路径

## 测试结果分析

### 成功指标
- **文档解析**：任务状态为 `completed`，无错误
- **向量构建**：索引文件生成成功，任务完成
- **检索功能**：返回相关结果，相似度分数合理
- **对话功能**：生成有意义的回复，响应时间合理

### 性能基准
- **文档解析**：< 30秒/文档
- **向量构建**：< 5分钟/1000个文档块
- **检索响应**：< 2秒
- **对话响应**：< 10秒

### 常见问题排查

1. **服务连接失败**
   - 检查服务是否启动
   - 验证端口是否被占用
   - 查看服务日志

2. **文档解析失败**
   - 检查文件格式是否支持
   - 验证文件是否损坏
   - 查看OCR配置

3. **向量构建失败**
   - 检查模型文件是否存在
   - 验证GPU/CPU资源
   - 查看内存使用情况

4. **检索结果质量差**
   - 调整chunk_size和chunk_overlap
   - 检查embedding模型
   - 优化查询文本

5. **对话响应异常**
   - 检查LLM配置
   - 验证API密钥
   - 查看模型负载

## 自定义测试

### 添加新的测试用例

1. 继承 `APITestCase` 类
2. 实现自定义测试方法
3. 添加到测试流程中

```python
class CustomTestCase(APITestCase):
    def test_custom_feature(self):
        # 自定义测试逻辑
        pass
```

### 修改测试参数

在脚本开头修改配置：

```python
# 服务地址
DOC_SERVICE_URL = "http://localhost:8000"
RAG_SERVICE_URL = "http://localhost:8001"

# 测试参数
TEST_TIMEOUT = 60
MAX_RETRIES = 3
```

### 集成到CI/CD

```yaml
# GitHub Actions 示例
- name: Run API Tests
  run: |
    python start_services.py &
    sleep 30
    python quick_test.py
    python test_workflow.py
```

## 性能测试

### 压力测试
使用 `test_api_cases.py` 进行并发测试：

```python
import concurrent.futures

def stress_test():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(tester.test_retrieval, index_id, query) 
                  for _ in range(100)]
        results = [f.result() for f in futures]
```

### 性能监控
测试过程中监控：
- CPU使用率
- 内存使用量
- GPU利用率
- 磁盘I/O
- 网络延迟

## 测试报告

所有测试脚本都会生成详细的测试报告：

- **JSON格式**：包含完整的测试数据和统计信息
- **控制台输出**：实时显示测试进度和结果
- **日志文件**：详细的执行日志

报告包含的信息：
- 测试开始和结束时间
- 每个步骤的执行时间
- 成功/失败统计
- 错误详情
- 性能指标

## 最佳实践

1. **测试前准备**
   - 确保系统资源充足
   - 清理之前的测试数据
   - 检查服务配置

2. **测试执行**
   - 按顺序执行测试
   - 监控系统资源
   - 记录异常情况

3. **结果分析**
   - 对比历史数据
   - 分析性能趋势
   - 识别潜在问题

4. **持续改进**
   - 根据测试结果优化配置
   - 更新测试用例
   - 完善错误处理
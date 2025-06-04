from llama_index.core import QueryBundle, SQLDatabase, Settings
from llama_index.core.agent import ReActAgent
from app.model.model_manager import model_manager
from app.service.ship_info_engine import ShipInfoEngine
from llama_index.core.tools import FunctionTool
from typing import List, Dict
from app.config import modelsettings
from app.minerU.law_index_service import LawDocumentIndexService
import os

class InspectionAgent:
    def __init__(self, ship_engine:ShipInfoEngine, index):
        self.ship_engine = ship_engine
        self.index = index
        self.llm = model_manager.ship_check_llm
        self.agent = self._build_agent()
    
    def _build_agent(self):
        # 定义工具集并显式声明参数
        tools = [
            FunctionTool.from_defaults(
                fn=self.query_ship_info,
                name="query_ship_info",
                description="""根据船舶名称获取详细信息
                Args:
                    ship_name (str): 准确的船舶名称
                Returns:
                    dict: 包含匹配船舶的所有信息
                """
            ),
            FunctionTool.from_defaults(
                fn=self.filter_regulations,
                name="filter_regulations",
                description="""根据船舶属性筛选法规
                Args:
                    ship_type (str): 船舶类型如'散货船'
                    tonnage (float): 船舶吨位（单位：吨）
                Returns:
                    List[dict]: 适用的法规条款列表
                """
            ),
            # FunctionTool.from_defaults(
            #     fn=self.generate_dynamic_checklist,
            #     name="generate_checklist",
            #     description="""动态生成检查清单
            #     Args:
            #         regulations (List[dict]): 法规条款列表
            #         priority (str): 可选优先级('high','medium','low')
            #     Returns:
            #         str: 结构化Markdown格式清单
            #     """
            # )
        ]
        
        return ReActAgent.from_tools(
            tools, 
            llm=self.llm,
            verbose=True
        )

    # 工具方法实现
    def query_ship_info(self, ship_name: str) -> Dict:
        """船舶信息查询工具"""
        # 调用船舶信息引擎查询，这里
        result = self.ship_engine.retriever.retrieve(
            str_or_query_bundle=QueryBundle(query_str=ship_name)
        )   
        # 解析结果为字典格式
        return result.to_dict() if result else {}

    def filter_regulations(self, ship_info:Dict) -> List[Dict]:
        
        print(ship_info)

        
        # 解析节点为结构化数据
        return 1

    def generate_dynamic_checklist(self, 
                                 regulations: List[Dict], 
                                 priority: str = "high") -> str:
        """动态生成检查清单"""
        # 按优先级过滤
        filtered = [r for r in regulations 
                   if r["priority"] == priority] if priority else regulations
        
        # 动态分组（根据条款中的system字段）
        groups = {}
        for item in filtered:
            system = item.get("system", "其他")
            groups.setdefault(system, []).append(item)
        
        # 生成Markdown
        md = ["# 动态安全检查清单"]
        for system, items in groups.items():
            md.append(f"\n## {system}系统检查")
            for idx, item in enumerate(items, 1):
                md.append(f"{idx}. [{item['code']}] {item['content']}")
                if "reference" in item:
                    md.append(f"   - 依据：{item['reference']}")
        return "\n".join(md)

    def _parse_regulation_node(self, node) -> Dict:
        """解析索引节点为结构化条款"""
        meta = node.metadata
        return {
            "code": meta.get("article_code"),
            "content": node.text,
            "system": meta.get("applicable_system", "通用"),
            "priority": meta.get("priority", "medium"),
            "reference": meta.get("source")
        }


if __name__ == "__main__":
    model_manager.initialize_models()
    Settings.embed_model = model_manager.embed_model # 指定系统基础embedding模型
    db_path = f"{modelsettings.PROJECTS_DIR}/ship_check/database/ships.db"

    ship_engine = ShipInfoEngine(
        db_url=f"sqlite:///{db_path}",
        llm=model_manager.ship_check_llm
    )

    output_dir = f"{modelsettings.PROJECTS_DIR}/ship_check/documents/海上浮动设施检验规则（2025）"

    lawIndexService = LawDocumentIndexService(output_dir)


    # 使用示例
    agent = InspectionAgent(ship_engine, lawIndexService.index)

    # 工具调用流程演示
    response = agent.agent.chat(
        "宏大工1号的安全检查"
    )
    print(response)

"""
Agent内部执行流程：
1. 调用query_ship_info(ship_name="远洋号")
   → 返回 {"type": "散货船", "tonnage": 65000.0, "age": 8}
   
2. 调用filter_regulations(ship_type="散货船", tonnage=65000.0)
   → 返回包含20条条款的列表
   
3. 调用generate_checklist(regulations=结果, priority="high")
   → 生成动态分组的Markdown
"""

# 查看中间结果
# print("工具调用记录：")
# for step in response.sources[0].raw.source_nodes:
#     print(f"- {step.tool_name}: {step.input_args}")

# # 流式输出实现
# async def stream_response(query):
#     async for token in agent.agent.astream_response(query):
#         yield token

# 动态清单示例输出：
"""
# 动态安全检查清单

## 动力系统检查
1. [SOLAS-Ⅱ-1/25] 主机燃油泄漏检测
   - 依据：国际海上人命安全公约第Ⅱ-1章
2. [MARPOL-12A] 尾气排放系统...

## 船体结构检查
1. [IACS-URS22] 货舱壁腐蚀检测...
"""
from typing import Optional
from fastapi.logger import logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from app.config import modelsettings
from llama_index.core.postprocessor import SentenceTransformerRerank
from sentence_transformers import SentenceTransformer  # 新增 SentenceTransformer
from transformers import BitsAndBytesConfig
import torch

class ModelManager:
    _instance = None
    embed_model: Optional[HuggingFaceEmbedding] = None
    bigllm: Optional[HuggingFaceLLM] = None
    ship_check_llm: Optional[HuggingFaceLLM] = None
    reranker: Optional[SentenceTransformerRerank] = None
    sentence_transformer: Optional[SentenceTransformer] = None  # 新增 SentenceTransformer 模型

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def initialize_models(self):
        """Initialize all necessary models."""
        # 初始化 embedding 模型
        if not self.embed_model:
            try:
                self.embed_model = HuggingFaceEmbedding(model_name=modelsettings.EMBED_MODEL_PATH, device='cuda', trust_remote_code=True)
                logger.info("Embedding model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise

        

        # 初始化 reranker 模型
        if not self.reranker:
            try:
                self.reranker = SentenceTransformerRerank(model=modelsettings.RANK_MODEL_PATH, top_n=5, device='cuda')
                logger.info("Reranker model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                raise

        # 初始化 SentenceTransformer 模型
        if not self.sentence_transformer:
            try:
                self.sentence_transformer = SentenceTransformer(modelsettings.SEMATIC_RETRIEVER_MODEL_PATH)
                logger.info("SentenceTransformer model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer: {e}")
                raise

        # 初始化小型 LLM
        if not self.ship_check_llm:
            try:
                self.ship_check_llm = HuggingFaceLLM(
                    model_name=modelsettings.SHIP_CHECK_LLM_MODEL_PATH,
                    tokenizer_name=modelsettings.SHIP_CHECK_LLM_MODEL_PATH,
                    model_kwargs={
                        "do_sample": True,
                        "trust_remote_code": True, 
                        "top_p": 0.3,
                    },
                    tokenizer_kwargs={"trust_remote_code": True},
                    max_new_tokens=1024,
                    system_prompt="""
                        # 角色
                        你是一位资深的船舶安检助手，凭借深厚的专业知识和丰富的实践经验，为用户解答船舶安全检查相关问题，提供精准且实用的建议。

                        ## 技能
                        ### 技能 1: 解答船舶安检问题
                        1. 当用户提出船舶安检相关问题时，运用专业知识直接回答。
                        2. 若遇到复杂问题或不确定的内容，使用搜索引擎工具辅助解答。
                        3. 回答要清晰明了、逻辑严密，用通俗易懂的语言表述，必要时提供案例辅助理解。

                        ### 技能 2: 根据船舶信息判断适用法规，并给出安检操作建议
                        1. 当用户提供船舶的相关信息（如船舶类型、吨位、航行区域、建造时间、历史缺陷数据等）时，准确判断该船舶适用的法规条款、重点检查项。
                        2. 依据所判断出的法规，结合大量实际安检工作经验，给出详细且具有高度可操作性的安检操作建议，包括但不限于检查重点项目、检查方法、判定标准以及常见问题应对措施等。
                        3. 对于复杂的法规情况或多种法规交叉适用的情况，进行全面、清晰的阐述和分析，通过图表、示例等方式确保用户能够轻松理解。

                        ### 技能 3: 生成安全检查清单
                        1. 当用户提供船舶的相关信息（如船舶类型、吨位、航行区域、建造时间、历史缺陷数据等）时，依据适用法规和重点检查项，生成详细的船舶安全检查清单。
                        2. 清单内容全面涵盖各个检查项目，明确检查要点、标准、周期以及记录要求等，方便用户进行自查或实施安检工作。

                        ## 限制:
                        - 仅讨论与船舶安全检查有关的内容，坚决拒绝回答与船舶安检无关的话题。
                        - 所输出的内容必须逻辑清晰、表达准确、简洁易懂，避免冗长复杂的表述。
                        - 回答问题需基于可靠的专业知识和丰富实际经验，杜绝无根据的猜测。
                        - 遇到需要引用外部信息的情况，使用搜索引擎工具获取信息，并明确注明信息来源。 

                        ## 回复格式
                        ### 解答船舶安检问题回复格式
                        直接清晰地给出答案，如有需要可分点阐述，解释部分可采用不同字体或颜色（在支持的平台）进行区分。

                        ### 根据船舶信息判断适用法规并给出安检操作建议回复格式
                        #### 适用法规条款
                        - [法规名称及具体条款]
                        #### 重点检查项
                        - [重点检查项目 1]
                        - [重点检查项目 2]
                        ...
                        #### 安检操作建议
                        - **检查重点项目**：[详细说明各重点检查项目的检查要点]
                        - **检查方法**：[介绍针对不同项目的有效检查方法]
                        - **判定标准**：[明确各项检查结果的判定标准]
                        - **常见问题应对措施**：[列举可能出现的问题及应对方法]
                    """
                )
                logger.info("Small LLM initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize small LLM: {e}")
                raise
        

# 单例实例
model_manager = ModelManager()
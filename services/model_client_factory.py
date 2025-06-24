from typing import Dict, Any, Optional
from config import EmbeddingModelSettings, LLMModelSettings, RerankModelSettings

# Import LlamaIndex LLM integrations
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic
from llama_index.llms.zhipuai import ZhipuAI as LlamaIndexZhipuAI
from llama_index.llms.siliconflow import SiliconFlow as LlamaIndexSiliconFlow
from llama_index.llms.deepseek import DeepSeek as LlamaIndexDeepSeek

# Import LlamaIndex Multi-Modal LLM integrations
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
# from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
# from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal
# from llama_index.multi_modal_llms.replicate import ReplicateMultiModal

# Import custom OpenAI-Like MultiModal implementation

# Import LlamaIndex Embedding integrations
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbedding
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding as LlamaIndexZhipuAIEmbedding
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding as LlamaIndexSiliconFlowEmbedding
# Assuming custom wrappers exist or need to be created for other providers
# from app.utils.baidu_llama_index import BaiduEmbedding
# from app.utils.siliconflow_llama_index import SiliconFlowEmbedding

# Import LlamaIndex Reranker integrations
from llama_index.core.postprocessor.types import BaseNodePostprocessor
# Assuming custom wrappers exist or need to be created for other providers
# from app.utils.cohere_rerank import CohereRerank
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

class ModelClientFactory:
    """模型客户端工厂类 - 基于LlamaIndex的统一模型接口"""
    
    @staticmethod
    def create_llm_client(model_config: LLMModelSettings) -> LLM:
        """创建LLM客户端 (使用LlamaIndex抽象)
        
        Args:
            model_config: 嵌入模型配置信息
            
        Returns:
            LLM: LlamaIndex LLM客户端实例
            
        Raises:
            ValueError: 不支持的模型提供商
        """
        provider_name = model_config.PROVIDER_NAME.lower()
        
        if provider_name == "openai":
            return LlamaIndexOpenAI(
                model=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL,
                temperature=model_config.TEMPERATURE,
                max_tokens=model_config.MAX_TOKENS,
                max_retries=model_config.MAX_RETRIES,
                system_prompt=model_config.SYSTEM_PROMPT
            )
        
        elif provider_name == "anthropic":
            return LlamaIndexAnthropic(
                model=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL,
                temperature=model_config.TEMPERATURE,
                max_tokens=model_config.MAX_TOKENS,
                max_retries=model_config.MAX_RETRIES,
                system_prompt=model_config.SYSTEM_PROMPT
            )
        elif provider_name == "zhipu":
            return LlamaIndexZhipuAI(
                model=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL,
                temperature=model_config.TEMPERATURE,
                max_tokens=model_config.MAX_TOKENS,
                max_retries=model_config.MAX_RETRIES,
                system_prompt=model_config.SYSTEM_PROMPT
            )
        elif provider_name == "siliconflow":
            print(model_config,'llm_model_config')
            return LlamaIndexSiliconFlow(
                model=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL,
                temperature=model_config.TEMPERATURE,
                max_tokens=model_config.MAX_TOKENS,
                max_retries=model_config.MAX_RETRIES,
                system_prompt=model_config.SYSTEM_PROMPT
            )
        elif provider_name == "deepseek":
            return LlamaIndexDeepSeek(
                model=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL,
                temperature=model_config.TEMPERATURE,
                max_tokens=model_config.MAX_TOKENS,
                max_retries=model_config.MAX_RETRIES,
                system_prompt=model_config.SYSTEM_PROMPT
            )
        else:
            #TODO 使用openailike代替
            raise ValueError(f"不支持的模型提供商: {provider_name}")
    
    @staticmethod
    def create_embedding_client(model_config: EmbeddingModelSettings) -> BaseEmbedding: 
        """创建嵌入模型客户端 (使用LlamaIndex抽象)
        
        Args:
            model_config: 模型配置信息
            
        Returns:
            BaseEmbedding: LlamaIndex嵌入模型客户端实例
            
        Raises:
            ValueError: 不支持的嵌入模型提供商
        """
        provider_name = model_config.PROVIDER_NAME.lower()
        
        if provider_name == "openai":
            return LlamaIndexOpenAIEmbedding(
                model_name=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL
            )
        
        elif provider_name == "siliconflow":
            return LlamaIndexSiliconFlowEmbedding(
                model_name=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL
            )
        elif provider_name == "zhipu":
            return LlamaIndexZhipuAIEmbedding(
                model_name=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
                api_base=model_config.API_BASE_URL
            )
        
        else:
            raise ValueError(f"不支持的嵌入模型提供商: {provider_name}")
    
    @staticmethod
    def create_rerank_client(model_config: RerankModelSettings) -> Optional[BaseNodePostprocessor]:
        """创建重排序模型客户端 (使用LlamaIndex抽象)
        
        Args:
            model_config: 重排序模型配置信息
            
        Returns:
            Optional[BaseNodePostprocessor]: LlamaIndex重排序模型客户端实例，如果不支持则返回None
        """
        provider_name = model_config.PROVIDER_NAME.lower()
        # TODO: Implement LlamaIndex wrappers for Cohere, SiliconFlow
        if provider_name == "siliconflow":
            return SiliconFlowRerank(
                model=model_config.MODEL_NAME,
                api_key=model_config.API_KEY,
            )
        
        else:
            raise ValueError(f"不支持的重排序模型提供商: {provider_name}")

    
    
   

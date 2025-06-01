

from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.llms import LLM
from llama_index.core.objects import ObjectIndex,SQLTableNodeMapping,SQLTableSchema
from llama_index.core import VectorStoreIndex
from sqlalchemy import create_engine, table

all_table_names = ['internal_ships',"domestic_ships"]

class ShipInfoEngine:
    def __init__(self, db_url: str,llm:LLM = None ):
        self.db_url = db_url
        self.llm = llm
        self._init_sql_database()
        self._init_sql_retriever()
        self._init_object_index()
        self.query_engine = self.create_query_engine()
    
    def _init_sql_database(self):
        engine = create_engine(self.db_url)
        self.database = SQLDatabase(engine, include_tables=all_table_names)

    def _init_sql_retriever(self):
        self.retriever = SQLRetriever(self.database)

    def _init_object_index(self):
        node_mapping = SQLTableNodeMapping(self.database)
        table_schema_objs = []
        # 获取数据库中所有表名
        for table_name in all_table_names:
            table_schema = SQLTableSchema(table_name=table_name)
            table_schema_objs.append(table_schema)
        self.object_index = ObjectIndex.from_objects(
            table_schema_objs, node_mapping
        )
        

    def create_query_engine(self):
        query_engine = SQLTableRetrieverQueryEngine(
            sql_database=self.database,
            table_retriever=self.object_index.as_retriever(similarity_top_k=1),
            llm=self.llm
        )
        return query_engine
    
    def query(self,query_str):
        response = self.query_engine.query(query_str)
        return response.response

    async def aquery(self,query_str):
        # 异步流式返回响应
        async for token in self.query_engine.aquery(query_str):
            yield token

    
        


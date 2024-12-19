import os
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import OpenAITextEmbedder




# 指定向量数据库chromaDB的存储位置和集合 根据自己的实际情况进行调整
CHROMADB_DIRECTORY = "chromaDB"  # chromaDB向量数据库的持久化路径
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称

# 加载环境变量参数
load_dotenv()

# 创建一个Chroma中的文档存储实例
document_store = ChromaDocumentStore(persist_path=CHROMADB_DIRECTORY,collection_name=CHROMADB_COLLECTION_NAME)

# 定义prompt模版 使用Jinja2 模板语法
prompt_template = """
    你是一个针对健康档案进行问答的机器人。
    你的任务是根据下述给定的已知信息回答用户问题。

    已知信息:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    用户问:
    {{question}}

    如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
    请不要输出已知信息中不包含的信息或答案。
    请不要输出已知信息中不包含的信息或答案。
    请不要输出已知信息中不包含的信息或答案。
    请用中文回答用户问题。
"""

# 设置调用 OpenAI Embedding模型 进行向量处理
# 这里注意，查询使用OpenAITextEmbedder
text_embedder = OpenAITextEmbedder(
    api_base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_EMBEDDING_MODEL")
)

# 设置调用 OpenAI Chat模型 生成内容
llm = OpenAIGenerator(
    api_base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_CHAT_MODEL")
)

# 用于从文档存储中根据查询找到最相关的文档
retriever = ChromaEmbeddingRetriever(document_store=document_store)

# 使用prompt模板构建自定义prompt
prompt_builder = PromptBuilder(template=prompt_template)

# 创建一个新的流水线对象
query_pipeline = Pipeline()

# 添加组件 name：组件名称 instance：组件实例
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.add_component("prompt_builder", prompt_builder)
query_pipeline.add_component("llm", llm)

# 连接组件
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "llm")

# 定义问题
question = "张三九的基本信息是什么"

# 运行流水线，并传入每个组件的初始输入
results = query_pipeline.run(
    data={
        "text_embedder": {"text": question},
        "retriever": {"top_k": 2},
        "prompt_builder": {"question": question},
    },
    include_outputs_from={"retriever","prompt_builder"}
)

# 运行结果，结果是一个嵌套字典
print(f"results:{results}\n")

# 从嵌套字典中取出最终结果
response = results["llm"]["replies"]
print(f"response:{response}\n")





import os
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder




# 加载环境变量参数
load_dotenv()

# 创建一个Chroma中的文档存储实例
document_store = ChromaDocumentStore(persist_path="ChromDB002")


# 灌入向量数据库
# 测试文档
documents=[
    Document(content="My name is Jean and I live in Paris.", meta={"title": "one"}),
    Document(content="My name is Mark and I live in Berlin.", meta={"title": "two"}),
    Document(content="My name is Giorgio and I live in Rome.", meta={"title": "three"})
]

# 写入向量数据库
writer = DocumentWriter(document_store)

# 设置调用 OpenAI Embedding模型 进行向量处理
# 这里注意，构建向量索引使用OpenAIDocumentEmbedder
index_embedder = OpenAIDocumentEmbedder(
    api_base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_EMBEDDING_MODEL")
)

# 创建一个新的流水线对象
indexing_pipeline = Pipeline()
# 添加组件 name：组件名称 instance：组件实例
indexing_pipeline.add_component("index_embedder", index_embedder)
indexing_pipeline.add_component("writer", writer)
indexing_pipeline.connect("index_embedder.documents", "writer.documents")

# 运行流水线，并传入每个组件的初始输入
results = indexing_pipeline.run(
    data={
        "index_embedder": {"documents": documents}
    },
    include_outputs_from={"writer"}
)
# 运行结果，结果是一个嵌套字典
print(f"results:{results}\n")


# 检索
# 定义prompt模版 使用Jinja2 模板语法
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
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
question = "Who lives in Paris?"

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





import os
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever




# 加载环境变量参数
load_dotenv()

# 创建一个Chroma中的文档存储实例
document_store = ChromaDocumentStore(persist_path="ChromDB001")


# 灌入向量数据库
# 测试文档
documents=[
    Document(content="My name is Jean and I live in Paris.", meta={"title": "one"}),
    Document(content="My name is Mark and I live in Berlin.", meta={"title": "two"}),
    Document(content="My name is Giorgio and I live in Rome.", meta={"title": "three"})
]

# 写入向量数据库
writer = DocumentWriter(document_store)

# 创建一个新的流水线对象
indexing_pipeline = Pipeline()
# 添加组件 name：组件名称 instance：组件实例
indexing_pipeline.add_component("writer", writer)
# 运行流水线，并传入每个组件的初始输入
results = indexing_pipeline.run(
    data={
        "writer": {"documents": documents}
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

# 用于从文档存储中根据查询找到最相关的文档
# ChromaQueryTextRetriever 使用默认的Embedding模型
# 模型所在位置:/Users/username/.cache/chroma/onnx_models/all-MiniLM-L6-v2
retriever = ChromaQueryTextRetriever(document_store=document_store)

# 使用prompt模板构建自定义prompt
prompt_builder = PromptBuilder(template=prompt_template)

# 设置调用 OpenAI Chat模型 生成内容
llm = OpenAIGenerator(
    api_base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_CHAT_MODEL")
)

# 创建一个新的流水线对象
query_pipeline = Pipeline()

# 添加组件 name：组件名称 instance：组件实例
query_pipeline.add_component("retriever", retriever)
query_pipeline.add_component("prompt_builder", prompt_builder)
query_pipeline.add_component("llm", llm)

# 连接组件
query_pipeline.connect("retriever", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "llm")

# 定义问题
question = "Who lives in Paris?"
# 运行流水线，并传入每个组件的初始输入
results = query_pipeline.run(
    data={
        "retriever": {"query": question, "top_k": 3},
        "prompt_builder": {"question": question},
    },
    include_outputs_from={"retriever","prompt_builder"}
)

# 运行结果，结果是一个嵌套字典
print(f"results:{results}\n")

# 从嵌套字典中取出结果
response = results["llm"]["replies"]
print(f"response:{response}\n")







import os
from dotenv import load_dotenv
from idlelib.rpc import response_queue
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret




# 加载环境变量参数
load_dotenv()

# 创建一个内存中的文档存储实例
# 文档将存储在内存中，不需要外部数据库支持
document_store = InMemoryDocumentStore()

# 向文档存储中添加文档
# 每个文档包含一段内容 content
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

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

# 基于 BM25 算法的检索器
# 用于从文档存储中根据查询找到最相关的文档
retriever = InMemoryBM25Retriever(document_store=document_store)

# 使用prompt模板构建自定义prompt
prompt_builder = PromptBuilder(template=prompt_template)

# 设置调用 OpenAI Chat模型 生成内容
# Secret用于安全地管理敏感信息
llm = OpenAIGenerator(
    api_base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
    # api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_CHAT_MODEL")
)

# 创建一个新的流水线对象
pipeline = Pipeline()

# 添加组件 name：组件名称 instance：组件实例
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)

# 连接组件
# retriever 的输出（相关文档列表）作为 prompt_builder.documents 的输入
pipeline.connect("retriever", "prompt_builder.documents")
# prompt_builder 的输出（生成的提示）作为 llm 的输入
pipeline.connect("prompt_builder", "llm")

# 定义问题
question = "Who lives in Paris?"

# 1、运行流水线，并传入每个组件的初始输入
results = pipeline.run(
    data={
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    },
    # include_outputs_from={"retriever","prompt_builder"}
)
# 运行结果，结果是一个嵌套字典
print(f"results:{results}\n")
# 从嵌套字典中取出结果
response = results["llm"]["replies"]
print(f"response:{response}\n")

# # 2、流水线可视化  保存为图片
# pipeline.draw(path="test.png")
#
# # 3、序列化 保存到YAML文件
# print(pipeline.dumps())
# with open("test.yml", "w") as file:
#   pipeline.dump(file)







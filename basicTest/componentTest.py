from typing import List
from haystack import component, Pipeline



# 声明这是一个 Haystack 组件，能够集成到 Pipeline 中
# 定义了一个类 生成个性化的欢迎信息并将其转为大写
@component
class WelcomeTextGenerator:
    # 定义组件的输出类型
    # welcome_text 是一个字符串，用于存储欢迎消息
    # note 是一个字符串，用于存储注释信息
    @component.output_types(welcome_text=str, note=str)
    # 定义组件的核心逻辑。name 是方法的输入参数，用于接受用户输入的名字
    # 返回一个字典，包含 welcome_text 和 note 两个键
    def run(self, name: str):
        return {"welcome_text": ('Hello {name}, welcome to Haystack!'.format(name=name)).upper(),
                "note": "welcome message is ready"}


# 声明这是一个 Haystack 组件，能够集成到 Pipeline 中
# 定义了一个类 根据空格拆分文本
@component
class WhitespaceSplitter:
    # 定义输出类型
    # splitted_text 是一个字符串列表，存储拆分后的文本
    @component.output_types(splitted_text=List[str])
    # 定义组件的核心逻辑。text 是方法的输入参数
    # 返回一个字典，包含键 splitted_text
    def run(self, text: str):
        return {"splitted_text": text.split()}


# 实例化一个Pipeline 对象
text_pipeline = Pipeline()
# 向流水线中添加组件
# name 是组件的标识符，用于引用组件
# instance 是组件的实例
text_pipeline.add_component(name="welcome_text_generator", instance=WelcomeTextGenerator())
text_pipeline.add_component(name="splitter", instance=WhitespaceSplitter())

# 连接流水线中的组件 将 welcome_text_generator 的输出字段 welcome_text 连接到 splitter 的输入字段 text
# sender 表示发送数据的组件和其输出字段
# receiver 表示接收数据的组件和其输入字段
text_pipeline.connect(sender="welcome_text_generator.welcome_text", receiver="splitter.text")

# 运行流水线
# 一个字典，用于为每个组件提供初始输入
result = text_pipeline.run({"welcome_text_generator": {"name": "Bilge"}})


# 运行结果，结果是一个嵌套字典
print(f"result:{result}\n")
# 从嵌套字典中取出结果
response = result["splitter"]["splitted_text"]
print(f"response:{response}\n")
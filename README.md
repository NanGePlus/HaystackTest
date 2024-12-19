# 1、项目介绍
**本次分享主要内容:**                      
(1)Haystack基础功能测试，包括4个测试demo脚本，分别演示其Pipeline功能、文件存储和检索功能、自定义组件功能                                                
(2)使用Haystack实现RAG应用(支持gpt大模型、国产大模型):                    
离线步骤:文档加载->文档切分->向量化->灌入向量数据库                     
在线步骤:获取用户问题->用户问题向量化->检索向量数据库->将检索结果和用户问题填入prompt模版->用最终的prompt调用LLM->由LLM生成回复                     
**关于RAG相关项目地址:**                                  
https://github.com/NanGePlus/RagLangChainTest                  

## 1.1 Haystack
### (1)Haystack定义   
Haystack 是一个开源框架，用于构建强大的问答（Question Answering, QA）、RAG等 AI 应用                       
它由 deepset 开发，旨在帮助开发者快速搭建基于自然语言处理（NLP）的信息查询和文档处理系统                       
Haystack 支持构建从小型本地化应用到大规模生产级应用的多种场景                                                                            
官方网址：https://haystack.deepset.ai/                                          
Github地址:https://github.com/deepset-ai/haystack             
### (2)向量数据库类型
向量库、纯向量数据库、支持向量的SQL数据库、支持向量的NoSQL数据库、全文搜索数据库                
<img src="./02.png" alt="" width="600" />                
### (3)流水线 Pipeline
pipeline是由不同组件编排生成的流水线，定义数据流转的逻辑，支持复杂的多组件流程                                           
<img src="./03.png" alt="" width="600" />

## 1.2 RAG定义及技术方案架构
### (1)RAG定义
RAG:Retrieval Augmented Generation(检索增强生成):通过使用检索的方法来增强生成模型的能力       
核心思想:人找知识，会查资料；LLM找知识，会查向量数据库        
主要目标:补充LLM固有局限性，LLM的知识不是实时的，LLM可能不知道私有领域业务知识          
场景类比:可以把RAG的过程想象成为开卷考试。让LLM先翻书查找相关答案，再回答问题              
### (2)技术方案架构        
技术架构图如下:                   
<img src="./01.png" alt="" width="600" />                     
离线步骤:文档加载->文档切分->向量化->灌入向量数据库                     
在线步骤:获取用户问题->用户问题向量化->检索向量数据库->将检索结果和用户问题填入prompt模版->用最终的prompt调用LLM->由LLM生成回复    
### (3)几个关键概念：
向量数据库的意义是快速的检索             
向量数据库本身不生成向量，向量是由Embedding模型产生的             
向量数据库与传统的关系型数据库是互补的，不是替代关系，在实际应用中根据实际需求经常同时使用         

## 1.3 Chroma
向量数据库，专门为向量检索设计的中间件                      


# 2、前期准备工作
## 2.1 开发环境搭建:anaconda、pycharm
anaconda:提供python虚拟环境，官网下载对应系统版本的安装包安装即可                                      
pycharm:提供集成开发环境，官网下载社区版本安装包安装即可                                               
**可参考如下视频:**                      
集成开发环境搭建Anaconda+PyCharm                                                          
https://www.bilibili.com/video/BV1q9HxeEEtT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                             
https://youtu.be/myVgyitFzrA          

## 2.2 大模型相关配置
**(1)GPT大模型使用方案**                             
**(2)非GPT大模型(国产大模型)使用方案(OneAPI安装、部署、创建渠道和令牌)**                                       
**(3)本地开源大模型使用方案(Ollama安装、启动、下载大模型)**                                               
**可参考如下视频:**                              
提供一种LLM集成解决方案，一份代码支持快速同时支持gpt大模型、国产大模型(通义千问、文心一言、百度千帆、讯飞星火等)、本地开源大模型(Ollama)                       
https://www.bilibili.com/video/BV12PCmYZEDt/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                 
https://youtu.be/CgZsdK43tcY           


# 3、项目初始化
## 3.1 下载源码
GitHub或Gitee中下载工程文件到本地，下载地址如下：                
https://github.com/NanGePlus/HaystackTest                                                               
https://gitee.com/NanGePlus/HaystackTest                                     

## 3.2 构建项目
使用pycharm构建一个项目，为项目配置虚拟python环境               
项目名称：HaystackTest                                                     

## 3.3 将相关代码拷贝到项目工程中           
直接将下载的文件夹中的文件拷贝到新建的项目目录中               

## 3.4 安装项目依赖                        
命令行终端中执行如下命令安装依赖包                               
pip install -r requirements.txt                      
每个软件包后面都指定了本次视频测试中固定的版本号                      
注意: 建议先使用要求的对应版本进行本项目测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                                
     

# 4、测试
## 4.1 基础功能测试   
### (1)运行python inMemoryPipelineTest.py                                         
使用内置的内存文档存储库(建议不能用于生产)进行测试                               
### (2)运行python chromaPipelineTest1.py                                      
使用chroma向量数据库并使用官方内置的Embedding模型进行测试                          
### (3)运行python chromaPipelineTest2.py                                         
使用chroma向量数据库并使用OpenAI(类OpenAI)的Embedding模型进行测试                     
### (4)运行python componentTest.py                                                                                      
Haystack自定义组件功能测试                             

## 4.2 RAG应用案例测试             
### (1)测试文档准备                                                          
这里以pdf文件为例，在input文件夹下准备了两份pdf文件                
健康档案.pdf:测试中文pdf文档处理                 
llama2.pdf:测试英文pdf文档处理                   
### (2)文本预处理后进行灌库                                                   
在tools文件夹下提供了pdfSplitTest_Ch.py脚本工具用来处理中文文档、pdfSplitTest_En.py脚本工具用来处理英文文档                        
vectorSaveTest.py脚本执行调用tools中的工具进行文档预处理后进行向量计算及灌库                           
使用python vectorSaveTest.py命令启动脚本                
### (3)检索测试            
首先运行python main.py命令启动脚本，开启API服务                                                           
再运行python apiTest.py命令进行POST请求                























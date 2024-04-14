# 1 茴香豆Web版问答领域助手

- 茴香豆是一个基于LLMs的领域知识助手，由书生浦语团队开发的开源大模型应用
- 专为即时通讯工具中的群聊场景优化的工作流，提供及时准确的技术支持和通过应用检索增强生成( RAG )术，能够理解和高效准确的回应与特定知识领域相关的复杂查询。
- 茴香豆使用了RAG技术，RAG( Retrieval Augmented Generation )是一种结合了检索( Retrieval) 和生成(Generation )的技术，旨在通过利用外部知识库来增强大型语言模型( LLMs )的性能。它通过检索与用户输入相关的信息片段，并结合这些信息来生成更准确、更丰富的回答。
- 特性

  - 开源免费：茴香豆使用 BSD-3-Clasue ，能够免费商用
  - 高效准确：茴香豆使用 Hybrid LLMs，优化群聊效果
  - 领域知识：应用RAG技术，来快速获取专业知识
  - 部署成本低：无需额外训练，可利云端模型 api，本地算力需求少
  - 安全：可完全本地本署，信息不上传至服务器，保护数据和用户隐私
  - 扩餐性强：兼容多种IM软件，支持多种开源LLMs和云端api
- 构成

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-010.png)

- 茴香豆工作流中为了解决即是通讯中可能会出现的灌水以及无关问题的情况，设计了拒回答机制，将会通过输入判断是否要回答问题，以提高应用效果

- 采用文档：python 3.12.3 官方说明 中文版

- 问答情况

  - print有什么用法

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-105.png)

  - for有什么用

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-106.png)

  - if有什么用

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-107.png)

  - list的用法

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-108.png)

  - 帮我写一份排序代码

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-109.png)

- 拒回答设置：设置拒绝回答“今天天气如何”问题设置

  - 问题设置：

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-110.png)

  - 使用效果：

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-111.png)

# 2 InternLM Studio 部署茴香豆助手

- 环境创建：Cuda11.7-conda 、30% A100 * 1
- RAG技术一项通过检索与用户输入相关的信息片段，并结合外部知识库来生成更准确、更丰富的回答。解决 LLMs 在处理知识密集型任务时可能遇到的挑战, 如幻觉、知识过时和缺乏透明、可追溯的推理过程等。提供更准确的回答、降低推理成本、实现外部记忆，RAG 技术的优势就是非参数化的模型调优，且具有以下特点：
  - 非参数记忆，利用外部知识库提供实时更新的信息。
  - 能够处理知识密集型任务，提供准确的事实性回答。
  - 通过检索增强，可以生成更多样化的内容
- 茴香豆是一个基于 LLM 的群聊知识助手
- 向量化数据库
  - 定义：将文本及其他数据通过其他预训练的模型转换为固定长度的向量表示，这些向量能够捕捉文本的语义信息。
  - 检索：根据用户的查询向量，使用向量数据库快速找出最相关的向量的过程通常通过计算余弦相似度或其他相似性度量来完成。检索结果根据相似度得分进行排序最相关的文档将被用于后续的文本生成。
  - 提取知识库特征，创建向量数据库。数据库向量化的过程应用到了 LangChain 的相关模块，默认嵌入和重排序模型调用的网易 BCE 双语模型
- 接受问题列表

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-101.png)

- 拒绝问题列表

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-100.png)

- 运行情况
  - 茴香豆怎么部署到微信群

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-102.png)

  - 今天天气怎么样？

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-103.png)

  - huixiangdou 是什么？

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-104.png)
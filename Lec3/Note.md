# 用茴香豆搭建RAG智能助理

## RAG

- 传统：收集语料 -> 微调
- RAG：不需要训练
- 定义：RAG( Retrieval Augmented Generation )是一种结合了检索( Retrieval) 和生成(Generation )的技术，旨在通过利用外部知识库来增强大型语言模型( LLMs )的性能。它通过检索与用户输入相关的信息片段，并结合这些信息来生成更准确、更丰富的回答。
- 通俗：搜索引擎，用户输入作为索引，通过大模型在数据搜索做出回答
-  图示

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-001.png)

- 传统RAG工作原理 索引 检索 生成

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/L%3ALM-LEC-003-002.png)

- 向量数据库

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-003.png)

- 示例

  - 1 检索向量化 2 检索 3 输入到生成模块生成回答

  ![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-004.png)

- 优化方法

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-005.png)

- 微调 vs RAG

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-006.png)

- 优化方法
  - 微调 提示工程 RAG

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-007.png)

## 茴香豆

- 茴香豆是一个基于LLMs的领域知识助手

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-008.png)

- 特性

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-009.png)

- 构建

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-010.png)

- 工作流
  - 拒答能够扩大应答助手的使用范围

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-011.png)

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC-003-013.png)


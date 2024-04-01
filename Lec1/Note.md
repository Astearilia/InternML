# 第一节 

- 大模型 等价于 语言建模
- 通用大模型：一个模型应对多种任务、多种模态

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC01000.png)

- 开源历程

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001001.png)

- InternLM2 体系

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001002.png)

- 新一代数据清洗过滤技术

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001003.png)

- 从模型到应用典型流程

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001004.png)

- 关键 语料 学习
- 书生·浦语全链条开源开放体系

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001005.png)

- 高质量语料数据

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001006.png)

- 微调

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001007.png)

- CompassKit 全栈工具链

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001008.png)

- 部署

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001009.png)

- 应用

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001010.png)

# Paper

## Abstract

- 本文讲的是 InternLM2 有优异表现
- 该模型在预处理时能够处理 text，code，long-context数据 
- 能够获得 long-term dependencies，从预训练的 4k token 到 32k token
- 使用了 Supervised Fine-Tuning(SFT)和 Reinforcement Learning form Human Feedback(RLHF) 来解决 conflicting 和 reward hacking

## Introduction

1. 本文介绍了 InternLM2，一种表现优异的大模型
2. 大语言模型主要有几个阶段
   1. 预训练 Pretraining
   2. 微调 有监督 Supervised Fine-Tuning(SFT)
   3. 强化学习 Reinforcement Learning form Human Feedback(RLHF)
3. 预训练
   1. 需要使用大量的语料
   2. 通过这个阶段来使得大模型具有广泛知识和基本技能
   3. 这个阶段的关键点是 **语料的质量**
   4. InternLM2 模型介绍了如何预训练文本
4. 拓展上下文长度
   1. 当前热点，许多下游应用都需要长的上下文
   2. InternLM2 做法：
      1. 使用 **Group Query Attention，GQA 组注意力机制**，可以实现更小的内存占用
      2. 论文里首先使用4K上下文文本，之后将语料转化为32K进行下一步训练
   3. 完成追后在通过 LocalLLaMA 机制取得很好成绩
5. **Supervised fine-tuning(SFT) +  Reinforcement Learning form Human Feedback(RLHF)**
   1. 使得模型能够很好的理解指令
   2. 通过构造32K数据提高InternLM2上下文能力
   3. 通过引入 **COOL RLHF**，这个模型通过多种reward model 来 reconcile 模型的conflict 和 diverse
   4. 通过 **Proximal Policy Optimization (PPO)** 来解决可能的 reward hacking 
6. 该模型的亮点有：在多个阶段表现优异，以及：
   1. 出色的性能
   2. 有200K的 Context Window
   3. 详尽的数据描述 包括各个阶段的
   4. 创新性的 RLHF 训练技术

## Infrastructure

### 1. InterEvo

- Framework 使用在 Pretraining ，SFT，和RLHF
- 这个模型是一个预训练框架
- 通过并行化data，tensor，sequence，pipeline来实现在多GPU上的训练
- 采用 ZeRO 来提高了GPU的内存效率，降低了 memory footprint
- 通过 FlashAttention 来提高 hardware utilization
- 效果
  - 在多 GPU 上训练具有很强的 scaling performance 
  - 有很强的 scaling of sequence length （25600 tokens）

#### Reducing Communication Overhead

- 解决问题：内存开销和通信开销之间的权衡
- 一般来说，communication cost 可以通过减少 communication scale 来减少
- 基于上述原则，可以通信限制在较少的一组GPU上，可能也同一node，进而降低总体的 communication cost
- 故InterEvo通过实现 adaptive sharding techniques 技术
  - 包含了 Full-Replica, Full-Sharding, and Partial-Sharding
  - parameters，gradients，optimizer states 能够通过硬件参数进行自选择

#### Communication-Computation Overlap

- 为了减communication overhead，InterEvo 协调 computation 和 communication 来降低
- 每个micro-batch 的 Forward 和 backward 过程中，通过 AllGather 开收集每一个 pre-fetches 的参数，通过该参数并行的计算当前层的 gradient。生成的 gradients 通过 ReduceScatter 在 parameter sharding group 组间进行 synchronization，通过AllReduce 在 parameter sharding groups 组中同步。这种方式提高 pipeline 的效率
- GPU 通过 Broad 更新parameters，InterEvo 通过 overlap 下一个 training 的 forward computation，平衡了communication overhead 和 computation time，提高了 performance

#### Long-Sequence Training

- 在这方面的问题是 long-sequence training 的computation speed 和 communication overhead 的权衡
- InternEvo 将GPU 的 memory 分解为一个 hierarchical space 的四个并行的方面 ： data， tensor，sequence，pipeline；以及三个 sharding ： parameter， gradient， optimizer state。
- 设计了一个根据 training scale，sequence length， model size，batch size 来定制 execution plan
- 实现了内存管理来解决碎片问题

#### Fault Tolerance

- GPU datacenter 训练 LLM 的问题，包括：硬件故障，并行计划的复杂性，资源利用效率等
- 团队研究了 LLM 和先前的 Deep learning 的 workloads 的区别，研究了 resource utilization pattern 和 job failure impacts
- 通过先前的研究得到两个 system efforts，
  - 一个 fault-tolerant pretraining system，能够自诊断和修复
  - 一个 decoupled scheduling system，评估task，提供performance feedback
- 引入了一个 asynchronous saving mechanism，定期存储model weight 和 optimizer state 到 distributed file 和 object storage 
- GPU 首先将数据保存到本地，在异步传输到系统
- 用处： 双异步过程保证了在系统检测到的网络以及硬件故障情况下，只损失极小的训练速度
- 将模型的 checkpoints 从热储存转移到冷储存，优化了储存空间
- 实现了在并行配置被更改的情况下的无缝训练，提高了训练的灵活性

#### Interactive Training

- 基于 Reinforcement Learning from Human Feedback (RLHF)，部署多个 LLM 进行交互训练

- 为了增强多个模型的 coordination ，开发了个新的 RLHF
  - 具有 flexibility 和 scalability 
  - 能够集成在多种 LLM execution engines 上，也支持多种 algorithmic designs

### Model Structure

- 为了提高效率，将 W_k,W_q,W_v，矩阵合并，提高了预训练效率
- 为了支持 多种 tensor parallelism ，不将 W_k,W_q,W_v 直接进行堆叠，而是采用交错的方式，这样就可以直接在 last dimension 进行划分，提高了分布式计算的灵活性
- 采用 Grouped-Query Attention (GQA)，因而可以在高速低GPU memory 情况下计算 long context

## Pre-train

### Pre-training data

- 解决问题：handling sensitive data, covering comprehensive knowledge, and balancing efficiency and quality、

#### Text Data

- 来源：web pages, papers, patents, and books
- 方法
  1. 判别来源，转换指定格式，通过 JSON Lines 储存 
  2. 再通过 rule-based filtering, data deduplication, safety filtering, and quality filtering 处理

##### Data Source Distribution

- 来源：中文，英文
- 形式：网页，书本，技术文档

##### Data Process Pipeline

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/InternLM2-000.png)

##### Data Formatting

1. 使用 Trafilatura 来 decompress，进行 HTML 解析和 main text extraction
2.  使用 pycld2 进行 language detection 和分别 main text
3. 分配 identifier ，并用 json 存储，得到 Format data

#####  Rule-based Stage

- web 数据中存在很多无效数据，例如错误等等
- 基本方法：regulation 和 filtering methods
- 设计了启发式过滤规则
  - 关注：标点，分隔的异常情况

##### Deduplication

- 重复数据影响大模型 training 效率
- 使用 Locality-Sensitive Hashing (LSH) 去重
- 使用 MinHash method (Broder, 1997) ，用 128 个哈希函数再 5-gram 文档上设置签名，使用0.7作为去重阈值
- 优先选择大的 CC dumps 来保存最近的数据

##### Safety Filtering

- 采用 “domain blocking”, “word blocking”, “pornography classifier”, and “toxicity classifier” 来过滤数据
- 进一步，通过 Kaggle 的 “Toxic Comment Classification Challenge” dataset 来 Fine-tune BERT 得到分类器

##### Quality Filtering

- web数据地址练练原因
  - 广告
  - 很多难以阅读的摘要以及说明
- 过滤器
  - 广告：人工标注
  - 流利度鉴别：一致性、噪音、信息内容和语法四个维度的评分
  - 通过标注的数据 fine-tuned BERT，得到过滤器，之后再进行二次过滤

#### Code Data

- 编码是LLM的关键技能
- 可以通过堆代码数据的训练增强模型的推理能力

#### Long Context Data

- 目前的热点之一，主要能够解决书籍摘要，长期对话，复杂推理
- 关键点在于长的上下文窗口

### Pre-training Settings

#### tokenization

- 采用GPT4的标记化方法
- 为了优化InternLM在处理中文文本时的压缩率，同时保持整体词汇量在10万以下，我们从cl100k词汇表中仔细挑选了前60，004个token，并将其与32，397个中文token集成。

#### Pre-training Hyper-parameters

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001100.png)

### Pre-training Phases

- 在第一阶段，我们使用了长度不超过4k的预训练语料。在第二阶段，我们纳入了50 %的长度不超过32k的预训练语料。在第三阶段，我们使用了特定能力的增强数据。在每个阶段，我们混合了英文、中文和代码中的数据。

## Alignment

- 两阶段 supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF).
- SFT 我们通过高质量的指令数据( Sec.4.1 )对模型进行微调，以遵循不同的人类指令。
- COOL RLHF 它应用了一种新颖的条件奖励模型，可以协调不同类型的偏好

### Supervised Fine-Tuning

- 我们使用了1000万条指令数据实例的数据集，这些数据实例经过筛选以确保它们的有用性和无害性。

### COOL Reinforcement Learning from Human Feedback

#### Conditional Reward Model

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001101.png)

- 条件奖励模型针对不同类型的偏好采用不同的系统提示，从而在一个单一的奖励模型中有效地建模多种偏好

#### Online RLHF

- 两种截然不同的路径：快速路径( Fast Path )和慢速路径( Slow Path )，
- 分别用于即时的、有针对性的改进和长期的、全面的奖赏模型优化。
- 快速路径和慢速路径是互补的，为减轻 Reward Hacking提供了一个自适应框架，并增强了用人类反馈训练的LLMs的性能和可靠性。

#### PPO Training Details

- 四个模型 the actor model, critic model, reference model,  reward model.
- 所有这些模型都具有相同的大小，确保了它们处理和生成数据的能力的一致性。

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001102.png)

### Long-Context Finetuning

- 两种类型的数据：一种是来自图书的长上下文数据，另一种是来自GitHub存储库的长上下文数据，并通过特定的范式进行拼接
- 为了增强InternLM2的数据分析能力，我们选择DS - 1000使用的代码库作为核心库
- 首先使用深度优先的方法对获取的原始数据进行排序
- 同时生成所需的提示，简要描述文件内容
- 随后，我们将处理后的数据依次拼接，直至达到32k的长度

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC001103.png)

## Conclusion

- 本文讲述的InternLM2模型在主客观方面都有很多优异表现
- 为了更好的支持 long context，模型使用 GQA 来降低 inference cost 
-  为了解决RLHF的preference conflict 使用了COOL RLHF


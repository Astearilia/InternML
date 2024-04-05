# 作业

## 基础作业

### 使用 `InternLM2-Chat-1.8B` 模型生成 300 字的小故事

截图

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002000.png)

文本

> 在一个美丽的花园里，住着一只名叫小花的蝴蝶。小花喜欢在花园里飞舞，追逐着花丛中的美丽花朵。有一天，小花的翅膀突然受伤了，她感到非常害怕和无助。她不知道该怎么办，只能停在花丛中哭泣。
>
> 就在这时，一只智慧的猫咪走过来，它看到了小花的样子，便走过去问小花：“小花，你怎么了？”小花回答说：“我的翅膀受伤了，我不知道该怎么办。”猫咪听了，立刻伸出爪子，轻轻地抚摸着小花受伤的翅膀，然后说道：“别担心，我会帮助你。”
>
> 猫咪开始帮助小花，它用柔软的舌头轻轻擦拭着小花受伤的翅膀，并用温暖的手轻轻地抚摸着小花，让它感到安心和舒适。小花渐渐地恢复了体力，恢复了飞行能力。
>
> 从那以后，小花和小猫成为了最好的朋友，小花学会了如何保护自己，并且帮助其他受伤的蝴蝶。小花和小猫的故事告诉我们，无论遇到什么困难，只要我们勇敢面对，寻求帮助，我们就能够克服困难，变得更加坚强和勇敢。

## 进阶作业

### `huggingface` 下载功能

代码

```python
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="internlm/internlm2-7b",
    filename="config.json",
    local_dir="./InternLM2-Chat-7B",
)

# 下载模型
os.system(
    "huggingface-cli download --resume-download internlm/internlm2-chat-7b --local-dir ./InternLM2-Chat-7B"
)
```

运行

- 由于网络原因只展示部分下载过程

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002001.png)

### `浦语·灵笔2` 的 `图文创作` 及 `视觉问答` 部署

#### 图文创作

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002002.png)

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002003.png)

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002004.png)

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002005.png)

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002006.png)

#### 视觉问答

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002007.png)

原图

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002008.jpg)

分析：原图是岩浆从黑色的岩石上流入大海，而不是模型认为的洞穴，且一般情况下洞穴不会向大海“泻入”

猜测是模型把黑色的玄武岩认作为洞穴而导致的错误

### `Lagent` 工具调用 `数据分析` Demo 部署

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002009.png)

![](https://astearilia.oss-cn-beijing.aliyuncs.com/PicGo/HPC/LLM-LEC002010.png)

可以看出模型顺利完成了数据计算任务
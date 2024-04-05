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

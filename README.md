<img src="YunCe_logo.png" alt="YunCe Logo" width="370" style="display: block; margin: 0 auto;">

# 项目介绍
YunCe (云策) 是一个基于 hello-agents 框架开发的, 针对数据科学竞赛的 "解读赛题-改进 baseline-验证有效性" 的自我更新 agent.

## 项目亮点
### 数据预处理部分

### 自动调优部分

### RAG 部分


# 项目安装
首先为 YunCe 创建虚拟环境: 
``python
uv add -r requirements.txt
``
> 若遇到版本不匹配问题, 优先下载 hello-agents 库. YunCe 能够在需要的时候自动安装其余库.

## 项目使用
1. 配置 system_prompt
- 设置其中 agent 的目标
- 
2. 根据自己的情况设置 .env 文件（需要将 ".env example" 修改为 ".env"）


## 适配不同赛题
- 将虚拟环境中的 helloagents1.0.0 库替换为本项目文件夹中的 hello_agents 文件夹
- 修改 project_config.py 中的路径
- 修改 policy.yaml 以增强本地安全性

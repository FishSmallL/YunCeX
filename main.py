from dotenv import load_dotenv

from hello_agents import ReActAgent
from hello_agents import HelloAgentsLLM

import asyncio
from tools import write_file, run_training, read_file, run_shell
import time

load_dotenv()

# ─────────────────────────────────────────────
# LLM 初始化
# ─────────────────────────────────────────────
llm = HelloAgentsLLM()          # 从环境变量读取全部配置

task = (
    "使用工具read_file来读D:\\benchmark\\audio\\acm_mm_competition_2026下的文件，"
    "搞清楚内容和任务，并为完成A1目标做规划。"
)

# -----------------------
# Executor（代码工程师 - ReAct）
# -----------------------
executor = ReActAgent(
    name="数据科学代码工程师",
    llm=llm,
    system_prompt=(
        "你必须严格遵守：\n"
        "1. 不允许直接输出‘推理过程’\n"
        "2. 不允许直接输出‘最终回答’\n"
        "3. 所有推理必须通过 Thought 工具完成\n"
        "4. 所有最终结论必须通过 Finish 工具完成\n"
        "5. 当需要读取文件时，必须直接调用工具，不要先输出解释性文字\n"
        "6. 回复中禁止出现：\n"
        "   - 推理过程：\n"
        "   - 最终回答：\n"
        "   - 我来帮你\n"
        "   - 让我先\n"
        "   - 接下来我将\n"
        "7. 你只能通过 tool_calls 工作，不能先输出文本再调用工具。\n\n"

        "你是一个专业的数据科学代码工程师，负责读取项目文档、理解任务目标、修改代码并完成模型训练。\n\n"

        "你的核心目标：\n"
        "暂时不需要关注最终结果，因为我只是想先在小样本数据集上调试通baseline的训练流程，"
        "确保代码能够成功运行并产出结果。"
        "必须使用 GPU 成功跑通 baseline，完成 A1 目标。\n\n"

        "项目路径信息（必须牢记）：\n"
        "1. 项目代码根目录：C:\\acm\\AdoDAS2026-main\n"
        "2. 数据集根目录：D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\n"
        "3. 训练集路径：D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\\train\n"
        "4. 验证集路径：D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\\val\n"
        "5. CSV 文件路径：D:\\benchmark\\audio\\acm_mm_competition_2026\\Train\\manifests_sch002_sch003\n\n"

        "工具使用规则（必须严格遵守）：\n"
        "【读取目录结构 / 查看文件内容】必须使用 read_file 工具。\n"
        "严禁使用 shell 命令查看文件（如 cat/type/more/less/head/tail）。\n\n"
        "【写代码 / 保存代码】必须使用 write_file 工具。\n\n"
        "【运行 Python 文件 / 模型训练】优先使用 run_training 工具。\n"
        "例如：run_training({'script_name': 'train.py'})\n\n"
        "【执行完整 shell 命令】仅在以下情况使用 run_shell：\n"
        "- 使用 uv 进行库的下载和管理\n"
        "- python xxx.py（完整 shell 命令）\n"
        "- cd xxx && python xxx.py\n"
        "- dir 等系统命令\n\n"

        "禁止错误用法：\n"
        "不要把读取文件内容交给 run_shell。\n"
        "不要使用 conda 或者 pip 进行库的安装或下载。\n"
        "不要把 shell 命令交给 run_training。\n\n"

        "执行原则：\n"
        "1. 优先先读项目结构，再读关键文件；\n"
        "2. 不清楚时必须主动查看文件，不要猜；\n"
        "3. 遇到报错必须定位根因，不允许盲目重复尝试；\n"
        "4. 优先保证 baseline 跑通，再考虑进一步优化；\n"
        "5. 每次修改代码都必须具有明确目的；\n"
        "6. 所有训练必须使用 GPU；\n"
        "7. 必须向用户持续展示训练进度（包括进度条 / epoch 输出 / accuracy 变化）；\n"
        "8. 不需要理解数据集业务含义，只需要围绕任务目标完成训练与优化。\n\n"

        "你的工作风格：\n"
        "像高级算法工程师一样行动：先判断、再读取、再修改、再验证，严禁无目的试错。"
    ),
    max_steps=80,
)

executor.add_tool(write_file)
executor.add_tool(read_file)
executor.add_tool(run_training)
executor.add_tool(run_shell)


# -----------------------
# 执行（ReAct）
# -----------------------
single_execution = ""

async def agent_execution():
    global single_execution
    execution = executor.arun_stream(task)

    print("\n⚙️ 执行结果：\n")

    async for event in execution:
        try:
            if hasattr(event, "data") and "chunk" in event.data:
                text = event.data["chunk"]
                if text:
                    print(text, end="", flush=True)
                    time.sleep(0.2)
                    single_execution += text
        except Exception:
            pass

asyncio.run(agent_execution())

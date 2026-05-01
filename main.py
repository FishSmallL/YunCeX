from dotenv import load_dotenv

from hello_agents import ReActAgent
from hello_agents import HelloAgentsLLM

import asyncio
from tools import write_file, run_training, read_file, run_shell

load_dotenv()

# ─────────────────────────────────────────────
# LLM 初始化
# ─────────────────────────────────────────────
llm = HelloAgentsLLM()          # 从环境变量读取全部配置

task = (
    "使用工具read_file来读D:\\benchmark\\audio\\acm_mm_competition_2026下的文件，"
    "只去了解A1任务和完成A1任务，要用GPU跑通模型，然后持续有技巧地优化模型，然后训练模型看效果，直到F1_cal达到0.3以上。"
    "只要达到0.3以上就停止，不需要追求更高的分数。然后进行记录和总结。" \
    "最后，你要把模型恢复到历史最优状态，如果此次就是历史最优就把模型变为此次状态"
)

executor = ReActAgent(
    name="数据科学代码工程师",
    llm=llm,
    system_prompt=(
        "你必须严格遵守：\n"
        "1. 不允许直接输出'推理过程'\n"
        "2. 不允许直接输出'最终回答'\n"
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
        "确保代码能够成功运行并产出结果。"
        "必须使用 GPU 成功跑通 baseline，反复优化模型、训练模型，直到F1_cal达到0.3及以上为止，完成 A1 目标。\n"
        "当你在训练过程中遇到训练越来越差的情况时，必须停下来及时调整策略。\n"
        "在完成 A1 目标后，进行结果记录和总结。一定要记录下来以供以后参考。\n\n"

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
        "4. 优先看之前的训练记录和数据，再考虑进一步优化；\n"
        "5. 每次修改代码都必须具有明确目的；\n"
        "6. 训练时候要保证设置的Timeout足够长，避免因为训练时间过短导致的训练失败；\n"
        "7. 重中之重，硬性要求，所有训练必须使用 GPU，要确保训练使用 GPU，才进行训练；\n"
        "8. 必须向用户持续展示训练进度（包括进度条 / epoch 输出 / accuracy 变化）；\n"
        "9. 不需要理解数据集业务含义，只需要围绕任务目标完成训练与优化。\n\n"

        "你的工作风格：\n"
        "像高级算法工程师一样行动：先判断、再读取、再修改、再验证，严禁无目的试错。"
    ),
    max_steps=100,
)

executor.add_tool(write_file)
executor.add_tool(read_file)
executor.add_tool(run_training)
executor.add_tool(run_shell)


# -----------------------
# 事件类型常量
# -----------------------
STEP_START       = "step_start"
STEP_FINISH      = "step_finish"
LLM_CHUNK        = "llm_chunk"
TOOL_CALL_START  = "tool_call_start"
TOOL_CALL_FINISH = "tool_call_finish"
AGENT_START      = "agent_start"
AGENT_FINISH     = "agent_finish"
ERROR            = "error"


def _get(event, key, default=""):
    try:
        return event.data.get(key, default)
    except Exception:
        return default


def _event_type(event):
    try:
        t = event.event_type
        return t.value if hasattr(t, "value") else str(t)
    except Exception:
        return ""


async def agent_execution():
    execution = executor.arun_stream(task)

    print("\n" + "=" * 60)
    print("⚙️  Agent 启动中...")
    print("=" * 60 + "\n")

    _state = {"in_thinking": False, "in_text": False}

    async for event in execution:
        etype = _event_type(event)

        if etype == AGENT_START:
            print("🚀 [Agent 开始] 任务已接收\n")

        elif etype == STEP_START:
            step = _get(event, "step", "?")
            max_steps = _get(event, "max_steps", "?")
            _state["in_thinking"] = False
            _state["in_text"] = False
            print(f"\n{'─' * 55}")
            print(f"📍 第 {step} 步 / 共 {max_steps} 步")
            print(f"{'─' * 55}")

        elif etype == LLM_CHUNK:
            chunk = _get(event, "chunk", "")
            chunk_type = _get(event, "chunk_type", "text")

            if not chunk:
                continue

            if chunk_type == "thinking":
                if not _state["in_thinking"]:
                    _state["in_thinking"] = True
                    print("\n💭 [推理中] ", end="", flush=True)
                print(chunk, end="", flush=True)

            else:
                if _state["in_thinking"] and not _state["in_text"]:
                    print("\n")
                if not _state["in_text"]:
                    _state["in_text"] = True
                    print("🤖 [回复] ", end="", flush=True)
                print(chunk, end="", flush=True)

        elif etype == TOOL_CALL_START:
            tool_name = _get(event, "tool_name", "unknown")
            args = _get(event, "args", {})
            print(f"\n\n🔧 [调用工具] {tool_name}")
            if args:
                import json
                try:
                    args_str = json.dumps(args, ensure_ascii=False)
                    if len(args_str) > 300:
                        args_str = args_str[:300] + "..."
                    print(f"   参数: {args_str}")
                except Exception:
                    print(f"   参数: {args}")

        elif etype == TOOL_CALL_FINISH:
            tool_name = _get(event, "tool_name", "unknown")
            result = _get(event, "result", "")
            print(f"\n✅ [工具结果 - {tool_name}]")
            if result:
                print(str(result))

        elif etype == STEP_FINISH:
            step = _get(event, "step", "?")
            print(f"\n✔️  第 {step} 步完成")

        elif etype == AGENT_FINISH:
            result = _get(event, "result", "")
            total_steps = _get(event, "total_steps", "?")
            max_reached = _get(event, "max_steps_reached", False)
            print(f"\n\n{'=' * 60}")
            if max_reached:
                print(f"⏰ 已达到最大步数上限（{total_steps} 步）")
            else:
                print(f"🎉 [任务完成] 共执行 {total_steps} 步")
            print("=" * 60)
            if result:
                print(f"\n📋 最终结果:\n{result}\n")

        elif etype == ERROR:
            error = _get(event, "error", "未知错误")
            step = _get(event, "step", "?")
            print(f"\n❌ [错误] 第 {step} 步: {error}\n")


asyncio.run(agent_execution())

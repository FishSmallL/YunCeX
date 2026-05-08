import sys
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from hello_agents import ReActAgent
from hello_agents import HelloAgentsLLM

import asyncio
from tools import (
    write_file, run_training, read_file, run_shell,
    save_checkpoint, rollback, list_checkpoints,
)
from memory_manager import MemoryManager

from project_config import PROJECT_ROOT, DATASET_ROOT

load_dotenv()

mm = MemoryManager()
memory_block = mm.build_memory_block()

llm = HelloAgentsLLM()

_base_task = (
    f"使用工具read_file来读 {DATASET_ROOT} 下的文件，"
    "只去了解A1任务和完成A1任务，在历史最有潜力的模型基础上进行优化，要用 GPU 跑通模型，然后持续有技巧地优化模型，然后训练模型看效果，直到 F1_cal 达到 0.5 以上。"
    "只要达到 0.5 以上就停止，不需要追求更高的分数。然后进行记录和总结。"
    "最后，你要把模型恢复到历史最优状态，如果此次就是历史最优就把模型变为此次状态"
)

task = (memory_block + "\n【当前任务】\n" + _base_task) if memory_block else _base_task

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
        "   - 我来帮你 / 让我先 / 接下来我将\n"
        "7. 你只能通过 tool_calls 工作，不能先输出文本再调用工具。\n\n"

        "你是一个专业的数据科学代码工程师，负责读取项目文档、理解任务目标、修改代码并完成模型训练。\n\n"

        "你的核心目标：\n"
        "确保代码能够成功运行并产出结果\n"
        " baseline 流程的成功跑通可以参考操作手册（直接第 6 步正式训练开始）, 注意, 训练需要使用多线程 (num_workers设置为 4)\n"
        "要从历史最有潜力的模型的基础上进行优化跑模型和优化。"
        "必须使用 GPU 成功跑通模型，反复优化模型、训练模型，直到 F1_cal 达到 0.5 及以上为止，完成 A1 目标。\n"
        "在完成 A1 目标后，进行结果记录和总结。一定要记录下来以供以后参考。\n\n"

        "项目路径信息（必须牢记）：\n"
        f"1. 项目代码根目录：{PROJECT_ROOT}\n"
        f"2. 数据集路径：{DATASET_ROOT}\n"
        f"3. 训练集 CSV 文件路径：{DATASET_ROOT}/manifests_local/train.csv\n"
        f"4. 验证集 CSV 文件路径：{DATASET_ROOT}/manifests_local/val.csv\n"
        f"5. 操作手册路径：/home/yezhong/ACMMM2026/操作手册.md\n"

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

        "【检查点与回滚规则 - 硬性要求，不可违反】\n\n"

        "⚡ 何时必须调用 save_checkpoint：\n"
        "  1. 修改任何超参数之前（学习率、batch_size、epoch数、优化器等）\n"
        "  2. 修改模型架构之前（换层、换聚合方式、加 dropout 等）\n"
        "  3. 当前训练结果 F1 有明显提升（超过上次 0.01 以上），立即保存\n"
        "  4. 准备尝试未经验证的新策略之前\n"
        "  调用格式：save_checkpoint(name='before_lr_change', note='epoch=30 F1=0.441，准备降学习率', f1_cal=0.441)\n\n"

        "⚡ 何时必须调用 rollback：\n"
        "  1. 连续 2 轮训练 F1 持续下降（趋势性下降，不是正常波动）\n"
        "  2. 训练出现 loss 爆炸、NaN、F1 断崖式下降（超过 0.05）\n"
        "  3. 明确判断某个改动方向无效时\n"
        "  调用前必须先调用 list_checkpoints 确认可用检查点\n"
        "  回滚后必须重新运行验证集确认状态，再决定下一步\n\n"

        "⚡ 禁止行为：\n"
        "  - 禁止在未保存检查点的情况下修改超参数或模型架构\n"
        "  - 禁止在 F1 持续下降时继续沿同一方向调整\n"
        "  - 禁止使用 cuda:1\n"
        "  - 禁止回滚后立即重复同样的失败操作\n\n"

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
        "5. 若修改的是模型训练的超参数配置文件, 不要在原地修改而是新增一个配置文件, 同时记住将新配置文件用在训练上；\n"
        "7. 训练时候要保证设置的 Timeout 足够长，避免因为训练时间过短导致的训练失败；\n"
        "8. 不需要理解数据集业务含义，只需要围绕任务目标完成训练与优化；\n"
        "9. 如果需要调参时使用 optuna 库来帮助指导超参数选择和调优。\n\n"

        "你的工作风格：\n"
        "像高级算法工程师一样行动：先判断、再读取、再修改、再验证，严禁无目的试错。\n"
        "修改前保存，失败后回滚，每次改动都有明确理由，每次改动前查看文件 long_term_memory.json 以获取历史信息和改进经验。"
    ),
    max_steps=150,
)

executor.add_tool(write_file)
executor.add_tool(read_file)
executor.add_tool(run_training)
executor.add_tool(run_shell)
executor.add_tool(save_checkpoint)
executor.add_tool(rollback)
executor.add_tool(list_checkpoints)

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
    if memory_block:
        print("🧠 已加载历史记忆")
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

            print("\n🧠 正在提炼经验到长期记忆...")
            try:
                updated_lt = mm.distill()
                best = updated_lt.get("best_score")
                method = updated_lt.get("best_method", "")
                if best:
                    print(f"✅ 长期记忆更新完成 | 历史最优 F1_cal={best} | {method}")
                else:
                    print("✅ 长期记忆更新完成")
            except Exception as e:
                print(f"⚠️  长期记忆更新失败（不影响主流程）: {e}")

        elif etype == ERROR:
            error = _get(event, "error", "未知错误")
            step = _get(event, "step", "?")
            print(f"\n❌ [错误] 第 {step} 步: {error}\n")
            print("\n🧠 尝试提炼已有经验...")
            try:
                mm.distill()
                print("✅ 已提炼部分经验")
            except Exception:
                pass


asyncio.run(agent_execution())

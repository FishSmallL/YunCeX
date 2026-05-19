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
    prepare_cleanlab_baseline, cleanlab_diagnose,
    apply_cleanlab_issue_fix, tabular_data_repair,
)
from data_quality_agent import build_data_quality_agent_tool
from memory_manager import MemoryManager

from project_config import PROJECT_ROOT, DATASET_ROOT, MEMORY_ROOT, ORIGINAL_PROJECT_ROOT, DEVICE
from agent_prompt import MAIN_AGENT_PROMPT

load_dotenv()

mm = MemoryManager()
memory_block = mm.build_memory_block()

llm = HelloAgentsLLM()

_base_task = (
    f"使用工具 read_file 来读 {DATASET_ROOT} 下的文件,"
    "只去了解A1任务和完成A1任务,"
    "调用 build_data_quality_agent_tool 工具去进行数据/特征质量诊断之后再进行训练/优化,"
    f"根据历史记录进行优化, 不仅要借鉴表现良好的模型, 也要在表现不好的方案中吸收经验教训. 使用 {DEVICE} 跑通模型"
    "要持续优化模型, 然后训练模型看效果, 直到 F1_cal 达到 0.5 以上."
    "注意, 最关键的是 test_metrics 中的 score_a1 指标 (计算方式为: score_a1= (f1_a+f1_s+f1_d)/3, 细节见 run_meta.json)!"
)

task = (memory_block + "\n【当前任务】\n" + _base_task) if memory_block else _base_task

executor = ReActAgent(
    name="数据科学代码工程师",
    llm=llm,
    system_prompt=MAIN_AGENT_PROMPT,
    max_steps=300,
)

executor.add_tool(write_file)
executor.add_tool(read_file)
executor.add_tool(run_training)
executor.add_tool(run_shell)
executor.add_tool(save_checkpoint)
executor.add_tool(rollback)
executor.add_tool(list_checkpoints)
executor.add_tool(
    build_data_quality_agent_tool(
        llm=llm,
        auxiliary_tools=[read_file, write_file, run_training, run_shell],
        default_data_loop_max_iterations=4,
    )
)

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
    import sys
    log_file_path = f"/home/yezhong/ACMMM2026/YunCe-main/logs/execution.log"  # 可以自定义路径，例如 f"{PROJECT_ROOT}/execution.log"

    class TeeStream:
        """一个将输出同时写入文件和原始标准输出的流"""
        def __init__(self, file, original_stdout):
            self.file = file
            self.original_stdout = original_stdout

        def write(self, data):
            # 写入文件
            self.file.write(data)
            self.file.flush()  # 确保及时写入磁盘
            # 输出到原终端
            self.original_stdout.write(data)
            self.original_stdout.flush()

        def flush(self):
            self.file.flush()
            self.original_stdout.flush()

    # 打开日志文件
    log_file = open(log_file_path, 'w', encoding='utf-8')
    # 保存原始标准输出
    original_stdout = sys.stdout
    # 创建Tee流并替换sys.stdout
    tee = TeeStream(log_file, original_stdout)
    sys.stdout = tee
    # ===== 新增代码结束 =====

    # 原有的executor.arun_stream(task)调用...
    execution = executor.arun_stream(task)

    print("\n" + "=" * 60)
    print(f"📁 日志同时保存至: {log_file_path}")  # 可选：在开头提示日志位置
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

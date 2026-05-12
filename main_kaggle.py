"""
演示主Agent通过Task工具调用KernelSkillAgent的完整流程。

用法:
  python main_kaggle.py                        # 全链路模拟 (ReActAgent → Task → KernelSkillAgent)
  python main_kaggle.py --direct               # 直接调用 arun_stream() 看纯流式事件
  python main_kaggle.py --direct --keyword "nlp"  # 自定义关键词
  python main_kaggle.py --keyword "computer_vision"  # 自定义关键词（全链路模式）

环境要求:
  - .env 文件中配置了 LLM_API_KEY 或 OPENAI_API_KEY
  - kaggle_knowledge/config.json 配置了 Kaggle 下载参数
"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
    print("错误: 未配置 LLM_API_KEY 或 OPENAI_API_KEY")
    print("请在项目根目录创建 .env 文件，参考 .env example")
    sys.exit(1)


# ── 解析命令行参数 ──
direct_mode = "--direct" in sys.argv
keyword = "deep_learning"
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--keyword" and i < len(sys.argv):
        keyword = sys.argv[i + 1]
        break

# ── 通用组件 ──
from hello_agents import HelloAgentsLLM, ReActAgent, Config
from hello_agents.core.streaming import StreamEvent, StreamEventType


def _chunk_str(event: StreamEvent) -> str:
    """从 StreamEvent 中提取可打印的文本内容"""
    chunk = event.data.get("chunk", "")
    return chunk


# ════════════════════════════════════════════════════════════════
# 模式 1: --direct  直接调用 arun_stream() 演示纯流式体验
# ════════════════════════════════════════════════════════════════

async def run_direct():
    from hello_agents.agents.kernel_skill_agent import KernelSkillAgent

    llm = HelloAgentsLLM()
    config = Config()

    project_root = Path(__file__).parent
    agent = KernelSkillAgent(
        name="kaggle-skill-extractor",
        llm=llm,
        config=config,
        skill_library_dir=str(project_root / "kaggle_knowledge" / "skill_library"),
        kernels_base_dir=str(project_root / "kaggle_knowledge" / "output"),
    )

    print(f"\n{'='*60}")
    print(f"  KernelSkillAgent 流式演示 (--direct 模式)")
    print(f"  关键词: {keyword}")
    print(f"{'='*60}")

    _in_text = False
    step_count = 0

    async for event in agent.arun_stream(keyword):
        etype = event.type

        if etype == StreamEventType.AGENT_START:
            print(f"\n🚀 [KernelSkillAgent] 启动\n")

        elif etype == StreamEventType.STEP_START:
            step_count += 1
            s = event.data.get("step", step_count)
            m = event.data.get("max_steps", "?")
            print(f"\n{'─'*55}")
            print(f"📍 步骤 {s} / {m}")
            print(f"{'─'*55}")
            _in_text = False

        elif etype == StreamEventType.STEP_FINISH:
            print(f"✔️  步骤完成")

        elif etype == StreamEventType.LLM_CHUNK:
            chunk = event.data.get("chunk", "")
            chunk_type = event.data.get("chunk_type", "text")
            if not chunk:
                continue
            if chunk_type == "thinking":
                if not _in_text:
                    print("\n💭 ", end="", flush=True)
                print(chunk, end="", flush=True)
            else:
                print(chunk, end="", flush=True)
            _in_text = True

        elif etype == StreamEventType.AGENT_FINISH:
            result = event.data.get("result", "")
            total = event.data.get("total_steps", step_count)
            print(f"\n\n{'='*60}")
            print(f"🎉 [KernelSkillAgent 完成] 共 {total} 步")
            print(f"{'='*60}")
            if result:
                print(f"\n📋 最终结果:\n{result}")

        elif etype == StreamEventType.ERROR:
            error = event.data.get("error", "未知错误")
            print(f"\n❌ [错误]: {error}")


# ════════════════════════════════════════════════════════════════
# 模式 2: 默认  全链路 ReActAgent → Task 工具 → KernelSkillAgent
# ════════════════════════════════════════════════════════════════

async def run_full_chain():
    from project_config import PROJECT_ROOT as _PR, DATASET_ROOT as _DR

    llm = HelloAgentsLLM()
    config = Config()

    # 创建主 Agent，提示词指引它使用 Task 工具
    main_agent = ReActAgent(
        name="数据科学主控Agent",
        llm=llm,
        config=config,
        system_prompt=(
            "你必须严格遵守：\n"
            "1. 所有推理通过 Thought 工具完成\n"
            "2. 所有最终结论通过 Finish 工具完成\n"
            "3. 你只能通过 tool_calls 工作\n\n"

            "你是一个数据科学主控Agent，负责调度子Agent完成特定任务。\n\n"

            "核心能力：\n"
            f"你需要提取 '{keyword}' 相关的 Kaggle 竞赛技巧。\n"
            f"请使用 Task 工具启动 kernel_skill 子Agent：\n"
            f"  Task(task='{keyword}', agent_type='kernel_skill')\n"
            "该子Agent会自动搜索Kaggle竞赛、下载高分kernel、"
            "用LLM提取可复用的ML技巧并保存到skill_library。\n\n"

            "执行步骤：\n"
            f"1. 调用 Task(task='{keyword}', agent_type='kernel_skill') 提取技巧\n"
            "2. 根据子Agent返回的结果，总结提取到了哪些技巧\n"
            "3. 调用 Finish 工具给出最终总结\n\n"

            "注意：\n"
            "- kernel_skill 子Agent 可能需要几分钟（涉及Kaggle下载和LLM提取）\n"
            "- 子Agent 返回后直接总结结果即可，不要重复调用\n"
        ),
        max_steps=10,
    )

    print(f"\n{'='*60}")
    print(f"  全链路演示: ReActAgent → Task 工具 → KernelSkillAgent")
    print(f"  关键词: {keyword}")
    print(f"{'='*60}")

    # 事件处理（复用 main.py 的模式）
    _state = {"in_thinking": False, "in_text": False}

    async for event in main_agent.arun_stream(
        f"请提取 '{keyword}' 相关的 Kaggle 竞赛技巧。"
    ):
        # 获取事件类型字符串
        try:
            t = event.type
            etype = t.value if hasattr(t, 'value') else str(t)
        except Exception:
            etype = ""

        if etype == "agent_start":
            print("\n🚀 [主Agent] 任务已接收\n")

        elif etype == "step_start":
            step = event.data.get("step", "?")
            max_s = event.data.get("max_steps", "?")
            _state["in_thinking"] = False
            _state["in_text"] = False
            print(f"\n{'─'*55}")
            print(f"📍 [主Agent] 第 {step} 步 / 共 {max_s} 步")
            print(f"{'─'*55}")

        elif etype == "llm_chunk":
            # ReActAgent 已在内部通过 print() 直接输出所有 chunk 内容，
            # 此处不重复打印，避免 UTF-8 多字节字符交错导致乱码。
            # 事件循环只负责结构标记（步骤、工具调用、完成等）。
            pass

        elif etype == "tool_call_start":
            tool_name = event.data.get("tool_name", "unknown")
            args = event.data.get("args", {})
            print(f"\n\n🔧 [主Agent 调用工具] {tool_name}")
            if args:
                import json
                try:
                    args_str = json.dumps(args, ensure_ascii=False)
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "..."
                    print(f"   参数: {args_str}")
                except Exception:
                    print(f"   参数: {args}")

        elif etype == "tool_call_finish":
            tool_name = event.data.get("tool_name", "unknown")
            result = event.data.get("result", "")
            print(f"\n✅ [工具结果 - {tool_name}]")
            if result:
                # 截断过长结果
                r = str(result)
                # if len(r) > 2000:
                #     r = r[:2000] + "\n...(已截断)"
                print(r)

        elif etype == "step_finish":
            step = event.data.get("step", "?")
            print(f"\n✔️  [主Agent] 第 {step} 步完成")

        elif etype == "agent_finish":
            result = event.data.get("result", "")
            total = event.data.get("total_steps", "?")
            max_reached = event.data.get("max_steps_reached", False)
            print(f"\n\n{'='*60}")
            if max_reached:
                print(f"⏰ [主Agent] 已达最大步数（{total} 步）")
            else:
                print(f"🎉 [主Agent 完成] 共 {total} 步")
            print(f"{'='*60}")
            if result:
                print(f"\n📋 最终结果:\n{result}")

        elif etype == "error":
            error = event.data.get("error", "未知错误")
            step = event.data.get("step", "?")
            print(f"\n❌ [主Agent 错误] 第 {step} 步: {error}")


# ════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════

def main():
    import asyncio

    if direct_mode:
        print(f"\n模式: --direct (直接调用 arun_stream)")
        asyncio.run(run_direct())
    else:
        print(f"\n模式: 全链路 (ReActAgent → Task → KernelSkillAgent)")
        print(f"提示: 此模式会消耗 LLM API tokens")
        print(f"      如果只想看流式效果，用 --direct 模式")
        input("\n按 Enter 开始，或 Ctrl+C 取消...")
        asyncio.run(run_full_chain())


if __name__ == "__main__":
    main()

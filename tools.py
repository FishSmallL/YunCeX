"""
tools.py — Agent 工具类
每个工具在执行前通过 policy.check(tool_name, target) 鉴权。
"""

import subprocess
import os
import threading
from typing import Dict, Any

from policy import policy  # 统一从策略单例鉴权
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode

from checkpoint_manager import checkpoint_manager  # ★ 检查点管理器


PROJECT_ROOT = r"C:\acm\AdoDAS2026-main"


def _resolve_path(path: str) -> str:
    return os.path.abspath(path)


class ReadFileTool(Tool):
    """读取文件或列出目录内容的工具。"""

    def __init__(self):
        super().__init__(name="read_file", description="读取文件或目录内容", expandable=False)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="file_path", type="string", description="文件或目录路径", required=True)
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path = parameters.get("file_path")
        if not file_path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 file_path"
            )

        if not policy.is_allowed_read(file_path):
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"read_file 无权访问: {file_path}"
            )

        path = _resolve_path(file_path)
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
                text = "\n".join(entries)
                return ToolResponse.success(
                    text=text or f"目录 '{file_path}' 为空",
                    data={"entries": entries}
                )
            except Exception as e:
                return ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"列目录失败: {e}"
                )

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return ToolResponse.success(
                text=content,
                data={"content": content}
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"读取失败: {e}"
            )


class WriteFileTool(Tool):
    """写文件工具。"""

    def __init__(self):
        super().__init__(name="write_file", description="将内容写入文件", expandable=False)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="file_path", type="string", description="目标文件路径", required=True),
            ToolParameter(name="content", type="string", description="要写入的内容", required=True)
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path = parameters.get("file_path")
        content = parameters.get("content")
        if not file_path or content is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 file_path 或 content"
            )

        if not policy.is_allowed_write(file_path):
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"write_file 无权写入: {file_path}"
            )

        path = _resolve_path(file_path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResponse.success(
                text=f"已写入: {path}",
                data={"path": path}
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"写入失败: {e}"
            )


class RunTrainingTool(Tool):
    """运行训练脚本的工具。"""

    def __init__(self):
        super().__init__(name="run_training", description="执行训练脚本", expandable=False)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="script_name", type="string", description="训练脚本名称", required=True),
            ToolParameter(name="timeout", type="integer", description="超时时长（秒）", required=False, default=7200)
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        script_name = parameters.get("script_name")
        timeout = parameters.get("timeout", 7200)
        if not script_name:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 script_name"
            )

        script_path = os.path.join(PROJECT_ROOT, script_name)
        if not policy.is_allowed_training(script_path):
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"run_training 无权执行: {script_path}"
            )

        try:
            process = subprocess.Popen(
                ["python", script_path],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            output_lines = []

            def _reader():
                if process.stdout is None:
                    return
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output_lines.append(line)

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()

            try:
                process.wait(timeout=int(timeout))
            except subprocess.TimeoutExpired:
                process.kill()
                reader_thread.join(timeout=1)
                return ToolResponse.error(
                    code=ToolErrorCode.TIMEOUT,
                    message=f"训练超时（{timeout}s）"
                )

            reader_thread.join(timeout=1)
            output = "".join(output_lines)
            return ToolResponse.success(
                text=output or "训练完成（无输出）",
                data={"returncode": process.returncode, "output": output}
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"训练失败: {e}"
            )


class RunShellTool(Tool):
    """执行 shell 命令的工具。"""

    def __init__(self):
        super().__init__(name="run_shell", description="执行 shell 命令", expandable=False)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="command", type="string", description="要执行的 shell 命令", required=True),
            ToolParameter(name="timeout", type="integer", description="超时时长（秒）", required=False, default=300)
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        command = parameters.get("command")
        timeout = parameters.get("timeout", 300)
        if not command:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 command"
            )

        if not policy.check("run_shell", command):
            return ToolResponse.error(
                code=ToolErrorCode.ACCESS_DENIED,
                message=f"run_shell 无权执行: {command}"
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=int(timeout),
            )
            output = result.stdout + result.stderr
            return ToolResponse.success(
                text=output or "命令执行完毕（无输出）",
                data={"returncode": result.returncode, "output": output}
            )
        except subprocess.TimeoutExpired:
            return ToolResponse.error(
                code=ToolErrorCode.TIMEOUT,
                message=f"命令超时（{timeout}s）"
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"命令执行失败: {e}"
            )


# ══════════════════════════════════════════════════════════════
# ★ 新增：检查点工具
# ══════════════════════════════════════════════════════════════

class SaveCheckpointTool(Tool):
    """
    保存当前模型为命名检查点。

    调用时机（Agent 必须遵守）：
    1. 修改任何超参数之前
    2. 当前 epoch F1 有明显提升时（如超过历史最优 0.01 以上）
    3. 准备尝试高风险改动前（换优化器、换架构等）

    示例调用：
        save_checkpoint(name="before_lr_decay", note="epoch=30 F1=0.441，准备降低学习率", f1_cal=0.441)
        save_checkpoint(name="best_smote_v2", note="BorderlineSMOTE调参后最优", f1_cal=0.461)
    """

    def __init__(self):
        super().__init__(
            name="save_checkpoint",
            description=(
                "保存当前模型为命名检查点，用于后续回滚。"
                "必须在修改超参数、换策略、高风险改动之前调用。"
            ),
            expandable=False
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description=(
                    "检查点名称，用英文或拼音，简明描述当前状态。"
                    "例如：before_lr_change / best_smote_v2 / epoch30_f1_441"
                ),
                required=True
            ),
            ToolParameter(
                name="note",
                type="string",
                description="备注：当前训练状态、F1分数、即将进行的操作，供回滚时参考",
                required=False,
                default=""
            ),
            ToolParameter(
                name="f1_cal",
                type="number",
                description="当前 F1_cal 分数（填写便于后续对比，可不填）",
                required=False,
                default=None
            ),
            ToolParameter(
                name="model_path",
                type="string",
                description=(
                    "要保存的模型文件路径（完整路径）。"
                    "不填则自动搜索 output 目录下最新的 .pt 文件。"
                ),
                required=False,
                default=None
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        name = parameters.get("name", "").strip()
        if not name:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 name，请提供检查点名称"
            )

        note = parameters.get("note", "")
        f1_cal = parameters.get("f1_cal")
        model_path = parameters.get("model_path") or None

        # f1_cal 类型保护
        if f1_cal is not None:
            try:
                f1_cal = float(f1_cal)
            except (TypeError, ValueError):
                f1_cal = None

        result = checkpoint_manager.save(
            name=name,
            model_path=model_path,
            note=note,
            f1_cal=f1_cal,
            trigger="agent",
        )

        if "error" in result:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=result["error"]
            )

        text = (
            f"✅ 检查点已保存\n"
            f"  名称: {result['name']}\n"
            f"  文件: {result['filename']}\n"
            f"  大小: {result['size_mb']} MB\n"
            f"  F1_cal: {result['f1_cal']}\n"
            f"  备注: {result['note']}\n"
            f"  时间: {result['saved_at']}\n"
            f"  路径: {result['path']}"
        )
        return ToolResponse.success(text=text, data=result)


class RollbackTool(Tool):
    """
    回滚到指定检查点。

    调用时机（Agent 必须遵守）：
    1. 连续 2 轮以上 F1 下降时，立即回滚到下降前保存的检查点
    2. 新策略导致训练崩溃（loss 爆炸 / F1 断崖式下降）时
    3. 明确判断某个改动无效，需要撤销时

    回滚后必须：
    1. 重新确认当前模型状态（跑一次验证集）
    2. 分析失败原因后再尝试新策略
    3. 不要在未保存检查点的情况下连续修改
    """

    def __init__(self):
        super().__init__(
            name="rollback",
            description=(
                "回滚模型到指定检查点。"
                "当训练效果持续变差或策略失败时必须调用，禁止盲目继续训练。"
            ),
            expandable=False
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description=(
                    "要回滚的检查点名称（支持模糊匹配，取最近一个匹配项）。"
                    "可先调用 list_checkpoints 查看所有可用检查点。"
                ),
                required=True
            ),
            ToolParameter(
                name="target_path",
                type="string",
                description=(
                    "回滚写入的目标模型路径（完整路径）。"
                    "不填则写回检查点原来的位置。"
                ),
                required=False,
                default=None
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        name = parameters.get("name", "").strip()
        if not name:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="缺少参数 name，请提供检查点名称"
            )

        target_path = parameters.get("target_path") or None

        result = checkpoint_manager.rollback(name=name, target_path=target_path)

        if "error" in result:
            # 回滚失败时，把可用检查点列表一起返回，方便 Agent 重新选择
            available = result.get("available_checkpoints", [])
            msg = result["error"]
            if available:
                msg += f"\n可用检查点：\n" + "\n".join(f"  - {n}" for n in available)
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=msg
            )

        text = (
            f"✅ 回滚成功\n"
            f"  回滚到: {result['rolled_back_to']}\n"
            f"  检查点时间: {result['checkpoint_saved_at']}\n"
            f"  检查点备注: {result['checkpoint_note']}\n"
            f"  检查点 F1_cal: {result['checkpoint_f1_cal']}\n"
            f"  写入路径: {result['target_path']}\n\n"
            f"⚠️  回滚完成后请重新运行验证集确认模型状态，再决定下一步策略。"
        )
        return ToolResponse.success(text=text, data=result)


class ListCheckpointsTool(Tool):
    """
    列出所有已保存的检查点。
    在决定回滚前调用，查看可用检查点及其 F1 分数。
    """

    def __init__(self):
        super().__init__(
            name="list_checkpoints",
            description="列出所有已保存的检查点，包含名称、F1分数、备注、保存时间。回滚前必须先调用此工具。",
            expandable=False
        )

    def get_parameters(self) -> list[ToolParameter]:
        return []  # 无需参数

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        checkpoints = checkpoint_manager.list_checkpoints()

        if not checkpoints:
            return ToolResponse.success(
                text="当前没有任何检查点。请在修改超参数前先调用 save_checkpoint 保存。",
                data={"checkpoints": []}
            )

        lines = [f"共 {len(checkpoints)} 个检查点（从新到旧）：\n"]
        for i, ckpt in enumerate(checkpoints, 1):
            f1_str = f"F1={ckpt['f1_cal']}" if ckpt.get("f1_cal") is not None else "F1=未记录"
            note_str = ckpt.get("note", "") or "无备注"
            lines.append(
                f"[{i}] {ckpt['name']}\n"
                f"     {f1_str} | 保存于 {ckpt['saved_at'][:19]}\n"
                f"     备注: {note_str}"
            )

        # 额外标注最优检查点
        best = checkpoint_manager.get_best_checkpoint()
        if best:
            lines.append(f"\n⭐ F1 最高检查点: {best['name']} (F1={best['f1_cal']})")

        text = "\n".join(lines)
        return ToolResponse.success(text=text, data={"checkpoints": checkpoints})


# ──────────────────────────────────────────────
# 兼容旧导出
# ──────────────────────────────────────────────
read_file        = ReadFileTool()
write_file       = WriteFileTool()
run_training     = RunTrainingTool()
run_shell        = RunShellTool()
save_checkpoint  = SaveCheckpointTool()   # ★ 新增
rollback         = RollbackTool()          # ★ 新增
list_checkpoints = ListCheckpointsTool()   # ★ 新增

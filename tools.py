"""
tools.py — Agent 工具类
每个工具在执行前通过 policy.check(tool_name, target) 鉴权。
"""

import subprocess
import os
from typing import Dict, Any

from policy import policy  # 统一从策略单例鉴权
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


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
            result = subprocess.run(
                ["python", script_path],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=int(timeout),
            )
            output = result.stdout + result.stderr
            return ToolResponse.success(
                text=output or "训练完成（无输出）",
                data={"returncode": result.returncode, "output": output}
            )
        except subprocess.TimeoutExpired:
            return ToolResponse.error(
                code=ToolErrorCode.TIMEOUT,
                message=f"训练超时（{timeout}s）"
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


# 兼容旧导出
read_file = ReadFileTool()
write_file = WriteFileTool()
run_training = RunTrainingTool()
run_shell = RunShellTool()

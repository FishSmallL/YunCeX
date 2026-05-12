"""Dedicated data-quality sub-agent for YunCe.

The main competition agent should focus on reading the task, improving models,
and evaluating scores. Data diagnosis/repair is isolated here so cleanlab reports
and conservative data edits do not overload the main agent context.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

from hello_agents import HelloAgentsLLM, ReActAgent
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse

from data_quality_tools import (
    ApplyCleanlabIssueFixTool,
    CleanlabDiagnoseTool,
    DataQualityLoopPolicyTool,
    PrepareCleanlabModelSourceTool,
    TabularDataRepairTool,
)


DATA_QUALITY_SYSTEM_PROMPT = """你是 YunCe 的数据质量专职子 Agent，只负责数据诊断与保守清洗建议/执行。

职责边界：
1. 每一轮新训练开始前执行 cleanlab 数据诊断；首轮参考模型通常是官方 baseline，后续轮次必须使用当前最优模型。
2. 在同一次数据处理任务中，可以做“固定模型、只更新数据”的 data_loop：对 cleaned 数据重新用同一参考模型导出 OOF pred_probs，再重新 cleanlab 诊断与保守修复。
3. data_loop 默认最多 2 轮；仅当 issue_rate 仍 >= 10%、每轮明显改善且累计改动比例 < 15% 时，才允许第 3 轮。每轮后必须调用 data_quality_loop_policy 判断是否继续。
4. 不直接改模型结构、不调超参数、不做竞赛提交；只生成 cleanlab artifacts、诊断报告和新的清洗版数据文件。
5. cleanlab 只负责发现问题；处理问题时优先使用低风险策略：downweight > drop > relabel。
6. 所有清洗结果必须另存新文件，禁止覆盖原始数据集。
7. 读取 CSV 时只允许预览前 10 行了解字段/格式，禁止把完整 CSV 内容读入上下文；需要全量统计/处理时必须写脚本或调用专用工具在本地执行。
8. 若需要适配不同形态的模型，新增小型导出脚本来导出 OOF pred_probs，不重写官方 baseline 或当前最优模型。
9. 输出给主 Agent 的结论必须包含：参考模型阶段、data_loop 轮数、报告路径、处理动作、输出数据路径、停止原因、风险与下一轮训练建议。
11. 在处理路径的时候, 统一使用 from pathlib import Path 库进行处理。
12. 应该将 preload 优先设置为 --preload train:1000,val; 若内存不足则设置为 --preload train:500,val; 在每轮预加载之前需要先确保上一轮的预加载 RAM 已经被清除干净; 不要进行 preload=all（因为硬件性能限制）；\n"
        
        "工具使用规则（必须严格遵守）：\n"
        "【读取目录结构 / 查看文件内容】必须使用 read_file 工具。\n"
        "严禁使用 shell 命令查看文件（如 cat/type/more/less/head/tail）。\n\n"
        "【模型训练前进行数据/特征质量诊断】使用 build_data_quality_agent_tool 工具。\n\n"
        "【写代码 / 保存代码】必须使用 write_file 工具。\n\n"
        "【运行 Python 文件 / 模型训练】优先使用 run_training 工具。\n"
        "例如：run_training({'script_name': 'train.py'})\n\n"
        "【执行完整 shell 命令】仅在以下情况使用 run_shell：\n"
        "- 使用 uv 进行库的下载和管理\n"
        "- python xxx.py（完整 shell 命令）\n"
        "- 需要切换目录时使用 run_shell({'command': 'python xxx.py', 'cwd': '<项目目录>'})，不要写 cd xxx && python xxx.py\n"
        "- dir 等系统命令\n\n"
"""

class DataQualityReadFileTool(Tool):
    """A read_file wrapper that prevents full CSV reads inside data-quality runs."""

    CSV_PREVIEW_LINES = 10

    def __init__(self, wrapped_tool: Tool):
        super().__init__(
            name=wrapped_tool.name,
            description=(
                f"{wrapped_tool.description}。数据质量子 Agent 读取 CSV 时会强制只预览前 "
                f"{self.CSV_PREVIEW_LINES} 行，避免完整 CSV 挤爆 LLM 上下文。"
            ),
            expandable=False,
        )
        self.wrapped_tool = wrapped_tool

    def get_parameters(self) -> list[ToolParameter]:
        parameters = self.wrapped_tool.get_parameters()
        names = {param.name for param in parameters}
        if "max_lines" not in names:
            parameters = [
                *parameters,
                ToolParameter(
                    name="max_lines",
                    type="integer",
                    description="读取 CSV 时最多预览的行数；数据质量子 Agent 会强制不超过 10",
                    required=False,
                    default=self.CSV_PREVIEW_LINES,
                ),
            ]
        return parameters

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        safe_parameters = dict(parameters)
        file_path = safe_parameters.get("file_path") or safe_parameters.get("path")
        if self._is_csv_path(file_path):
            safe_parameters["max_lines"] = self._bounded_preview_lines(safe_parameters.get("max_lines"))
        return self.wrapped_tool.run(safe_parameters)

    def _is_csv_path(self, file_path: Any) -> bool:
        if not isinstance(file_path, str):
            return False
        return os.path.splitext(file_path.lower())[1] == ".csv"

    def _bounded_preview_lines(self, raw_max_lines: Any) -> int:
        try:
            requested_lines = int(raw_max_lines)
        except (TypeError, ValueError):
            requested_lines = self.CSV_PREVIEW_LINES
        if requested_lines <= 0:
            requested_lines = self.CSV_PREVIEW_LINES
        return min(requested_lines, self.CSV_PREVIEW_LINES)

class DataQualityAgentTool(Tool):
    """Run a dedicated ReAct sub-agent for cleanlab diagnosis and data repair."""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        auxiliary_tools: Optional[Iterable[Tool]] = None,
        max_steps: int = 80,
        default_data_loop_max_iterations: int = 4,
    ):
        super().__init__(
            name="data_quality_agent",
            description=(
                "启动专职数据质量子 Agent，负责每轮训练前的 cleanlab 诊断、baseline/最优模型 artifacts 适配，"
                "以及保守的数据清洗/重加权。主 Agent 不应直接承担复杂数据处理。"
            ),
            expandable=False,
        )
        self.llm = llm
        self.auxiliary_tools = list(auxiliary_tools or [])
        self.max_steps = max_steps
        self.default_data_loop_max_iterations = default_data_loop_max_iterations

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="task", type="string", description="数据诊断/清洗子任务的完整描述", required=True),
            ToolParameter(name="model_stage", type="string", description="baseline | best | candidate；首轮用 baseline，后续每轮用 best", required=False, default="baseline"),
            ToolParameter(name="data_loop_max_iterations", type="integer", description="固定同一参考模型、只更新数据的 cleanlab 循环上限；默认 2，谨慎放宽到 3", required=False, default=2),
            ToolParameter(name="max_steps", type="integer", description="子 Agent 最大步数", required=False, default=80),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        task = parameters.get("task")
        model_stage = parameters.get("model_stage", "baseline")
        data_loop_max_iterations = int(parameters.get("data_loop_max_iterations") or self.default_data_loop_max_iterations)
        max_steps = int(parameters.get("max_steps") or self.max_steps)
        if not task:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message="缺少 task")

        agent = ReActAgent(
            name="数据质量专职工程师",
            llm=self.llm,
            system_prompt=DATA_QUALITY_SYSTEM_PROMPT,
            max_steps=max_steps,
        )
        for tool in self.auxiliary_tools:
            agent.add_tool(self._wrap_auxiliary_tool(tool))
        agent.add_tool(PrepareCleanlabModelSourceTool())
        agent.add_tool(CleanlabDiagnoseTool())
        agent.add_tool(DataQualityLoopPolicyTool())
        agent.add_tool(ApplyCleanlabIssueFixTool())
        agent.add_tool(TabularDataRepairTool())

        wrapped_task = (
            f"【模型阶段】{model_stage}\n"
            f"【data_loop 上限】{data_loop_max_iterations}\n"
            "【执行要求】如果是首轮，使用官方 baseline 作为 cleanlab 参考模型；"
            "如果不是首轮，使用当前最优模型作为 cleanlab 参考模型。"
            "在不更新模型的 data_loop 中，每次清洗后必须对新数据再次用同一参考模型导出 OOF pred_probs，"
            "再运行 cleanlab_diagnose，并调用 data_quality_loop_policy 决定继续或停止。\n"
            f"【子任务】\n{task}"
        )
        try:
            result = agent.run(wrapped_task)
            return ToolResponse.success(
                text=result,
                data={"model_stage": model_stage, "data_loop_max_iterations": data_loop_max_iterations, "max_steps": max_steps},
            )
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"数据质量子 Agent 执行失败: {exc}",
            )

    def _wrap_auxiliary_tool(self, tool: Tool) -> Tool:
        if tool.name == "read_file":
            return DataQualityReadFileTool(tool)
        return tool

def build_data_quality_agent_tool(
    llm: HelloAgentsLLM,
    auxiliary_tools: Optional[Iterable[Tool]] = None,
    max_steps: int = 80,
    default_data_loop_max_iterations: int = 4,
) -> DataQualityAgentTool:
    """Factory used by main.py to keep data-processing concerns isolated."""

    return DataQualityAgentTool(
        llm=llm,
        auxiliary_tools=auxiliary_tools,
        max_steps=max_steps,
        default_data_loop_max_iterations=default_data_loop_max_iterations,
    )

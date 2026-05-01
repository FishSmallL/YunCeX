import os
import ast
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from hello_agents import ToolRegistry
from hello_agents.tools import Tool, ToolErrorCode, ToolParameter, ToolResponse
from policy import policy


# ══════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════

# 训练/Shell 输出截断上限（日志类，3000 字符足够定位问题）
MAX_LOG_OUTPUT_CHARS = 3_000

# 文件读取返回给 LLM 的最大字符数。
# 根源问题：原来 SmartReadFileTool 的 max_chars 默认值是 8000，
# 但 react_agent 的 self.truncator.truncate() 会在工具返回后
# 再做一次截断，其默认阈值往往只有 2000~4000 字符，
# 两层截断叠加导致代码文件几乎必然被切断。
# 解决方案：
#   1. 把文件工具自身的 max_chars 提升到 20000（覆盖绝大多数代码文件）
#   2. 新增 chunk_read 分段读取，文件太长时 agent 可以按段读取
#   3. 在工具输出的 text 末尾附加行号信息，agent 可精确定位后续分段
MAX_FILE_READ_CHARS = 20_000   # 单次读取上限（约 500-800 行代码）
MAX_FILE_CHUNK_CHARS = 8_000   # 分段读取每段上限

# run_training / run_shell 的默认超时（秒）
DEFAULT_TIMEOUT = 600


# ══════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════

def _truncate_log(text: str, max_chars: int = MAX_LOG_OUTPUT_CHARS) -> str:
    """
    截断训练/shell 日志输出：保留头部摘要 + 尾部完整错误。
    日志类输出 3000 字符已足够 agent 定位问题，避免上下文膨胀。
    """
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head - 50
    return (
        text[:head]
        + f"\n\n... [日志过长，已截断中间 {len(text) - max_chars} 字符，保留头尾] ...\n\n"
        + text[-tail:]
    )





def extract_python_code(text: str) -> str:
    """从文本中提取 Python 代码块（支持 markdown / 三引号 / 纯代码）"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("❌ 输入为空，无法提取代码")

    def normalize(code: str) -> str:
        code = code.strip()
        code = re.sub(r"^\s*(python|py)\s*\n", "", code, flags=re.I)
        return code.strip()

    candidates: List[str] = []

    md_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.S | re.I)
    if md_blocks:
        candidates.append("\n\n".join(normalize(b) for b in md_blocks if normalize(b)))

    triple_blocks: List[str] = []
    triple_blocks += re.findall(r'"""(.*?)"""', text, re.S)
    triple_blocks += re.findall(r"'''(.*?)'''", text, re.S)
    if triple_blocks:
        candidates.append("\n\n".join(normalize(b) for b in triple_blocks if normalize(b)))

    candidates.append(normalize(text))

    uniq: List[str] = []
    for c in candidates:
        if c and c not in uniq:
            uniq.append(c)

    for code in uniq:
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue

    if uniq:
        return max(uniq, key=len)

    raise ValueError("❌ 未找到可用代码")


def validate_python(code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"❌ Python 语法错误: {e}") from e


def _run_subprocess(
    cmd: List[str],
    timeout: int,
    cwd: Optional[str] = None,
    shell: bool = False,
) -> Dict[str, Any]:
    """
    通用子进程执行器（run_training / run_shell 共用）。

    实时输出：
    - 用 select 同时监听 stdout + stderr，逐行读取并立即 print，
      让训练进度条/epoch 日志实时显示在终端。
    - 同时把所有输出收集起来，最终截断后返回给 LLM。
    - 超时后 kill() + wait() 确保子进程彻底退出。
    """
    import select
    import sys
    import threading

    try:
        process = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            bufsize=1,          # 行缓冲，配合 -u 让 Python 子进程实时 flush
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        timed_out = False

        # Windows 不支持 select 对 PIPE，用线程方案统一处理
        def _reader(pipe, collector):
            # 逐字符读，正确处理 \r（tqdm进度条）和 \n（普通日志）
            buf = []
            while True:
                ch = pipe.read(1)
                if not ch:
                    # 管道关闭，输出剩余缓冲
                    if buf:
                        line = "".join(buf)
                        collector.append(line + "\n")
                        print(line, flush=True)
                    break
                if ch == "\r":
                    line = "".join(buf)
                    buf = []
                    collector.append(line + "\n")
                    # \r 用 end="\r" 覆盖同一行（tqdm进度条效果）
                    print(line, end="\r", flush=True)
                elif ch == "\n":
                    line = "".join(buf)
                    buf = []
                    collector.append(line + "\n")
                    print(line, flush=True)
                else:
                    buf.append(ch)
            pipe.close()

        t_out = threading.Thread(target=_reader, args=(process.stdout, stdout_lines), daemon=True)
        t_err = threading.Thread(target=_reader, args=(process.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            timed_out = True

        # 等读取线程完成
        t_out.join(timeout=5)
        t_err.join(timeout=5)

        return_code = process.returncode if not timed_out else -1

        combined = "".join(stdout_lines).strip()
        if stderr_lines:
            stderr_text = "".join(stderr_lines).strip()
            if stderr_text:
                combined += "\n\n[stderr]\n" + stderr_text

        return {
            "success": return_code == 0,
            "output": _truncate_log(combined),
            "return_code": return_code,
            "timed_out": timed_out,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "output": f"❌ 命令未找到: {e}",
            "return_code": -1,
            "timed_out": False,
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"❌ 执行异常: {e}",
            "return_code": -1,
            "timed_out": False,
        }


def _parse_metrics(output: str) -> Dict[str, Any]:
    """
    从训练输出中解析关键指标。

    根源问题修复：原来只解析 "Final Accuracy"，
    但任务目标是 F1_cal >= 0.3，根本解析不到，
    agent 永远看到 accuracy=0.0，无法判断是否达标。
    """
    metrics: Dict[str, Any] = {}

    # F1_cal（任务核心指标）
    m = re.search(r"[Ff]1[_\-]?[Cc]al(?:ibration)?[:\s=]+([0-9]+\.?[0-9]*)", output)
    if m:
        metrics["f1_cal"] = float(m.group(1))

    # F1（通用）
    m = re.search(r"\b[Ff]1[_\-][Ss]core[:\s=]+([0-9]+\.?[0-9]*)", output)
    if m:
        metrics["f1_score"] = float(m.group(1))

    # Accuracy / Final Accuracy
    m = re.search(r"(?:Final\s*)?[Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)", output)
    if m:
        metrics["accuracy"] = float(m.group(1))

    # Loss（最后一次出现）
    losses = re.findall(r"[Ll]oss[:\s=]+([0-9]+\.?[0-9]*)", output)
    if losses:
        metrics["last_loss"] = float(losses[-1])

    # Epoch（最后一次出现）
    epochs = re.findall(r"[Ee]poch\s*[\[/]?\s*(\d+)", output)
    if epochs:
        metrics["last_epoch"] = int(epochs[-1])

    return metrics


# ══════════════════════════════════════════════════════════════
# Tool 1：写文件
# ══════════════════════════════════════════════════════════════

class WriteFileTool(Tool):
    """
    将代码内容写入指定文件。

    改进：
    - output_file 改为运行时参数（不再硬编码 result.py），
      agent 可以直接指定目标路径（如 C:\\acm\\AdoDAS2026-main\\train.py）。
    - 自动创建父目录，避免因目录不存在而失败。
    - 支持写入任意文本文件（不仅限于 .py），满足写 config/yaml 等需求。
    """

    def __init__(self):
        super().__init__(
            name="write_file",
            description=(
                "将代码或文本内容写入文件。"
                "自动进行 Python 语法校验（仅 .py 文件）。"
                "output_file 须为完整路径，例如 C:\\acm\\project\\train.py"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="content",
                type="string",
                description="要写入的代码或文本内容",
                required=True,
            ),
            ToolParameter(
                name="output_file",
                type="string",
                description="目标文件完整路径，例如 C:\\acm\\AdoDAS2026-main\\train.py",
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        content = parameters.get("content") or parameters.get("input")
        output_file = parameters.get("output_file", "").strip()

        if not isinstance(content, str) or not content.strip():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="❌ 缺少参数 content（不能为空）",
            )

        if not output_file:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="❌ 缺少参数 output_file（目标文件路径不能为空）",
            )

        if not policy.is_allowed_write(output_file):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"❌ 禁止写入文件：{output_file}（不在写入白名单）",
            )

        try:
            # 对于 .py 文件，先尝试提取并校验语法
            if output_file.endswith(".py"):
                try:
                    code = extract_python_code(content)
                    validate_python(code)
                except ValueError as e:
                    return ToolResponse.error(
                        code=ToolErrorCode.INVALID_FORMAT,
                        message=str(e),
                    )
            else:
                code = content.strip()

            # 自动创建父目录
            parent_dir = os.path.dirname(output_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code)

            return ToolResponse.success(
                text=f"✅ 已写入文件：{output_file}（{len(code)} 字符）",
                data={"output_file": output_file, "size": len(code)},
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"❌ 写文件失败：{e}",
            )


# ══════════════════════════════════════════════════════════════
# Tool 2：运行训练脚本
# ══════════════════════════════════════════════════════════════

class RunTrainingTool(Tool):
    """
    执行 Python 训练脚本并返回结果。

    改进：
    - 支持 working_dir 参数，解决相对路径引用资源失败的问题。
    - 支持 extra_args 传递命令行参数（如 --epochs 30 --lr 0.001）。
    - 解析 F1_cal / F1 / Accuracy / Loss，不再只看 Final Accuracy。
    - 超时使用 communicate(timeout=) 代替 readline() 轮询，更可靠。
    - stderr 单独捕获并附加在输出末尾，traceback 不再丢失。
    - 输出截断策略：保留头部摘要 + 尾部（错误在尾部）。
    """

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT):
        super().__init__(
            name="run_training",
            description=(
                "执行 Python 训练脚本，返回训练输出和关键指标（F1_cal、accuracy、loss 等）。"
                "script_path 须为完整路径或相对于 working_dir 的路径。"
            ),
        )
        self.default_timeout = default_timeout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="script_path",
                type="string",
                description="训练脚本完整路径，例如 C:\\acm\\AdoDAS2026-main\\train.py",
                required=True,
            ),
            ToolParameter(
                name="working_dir",
                type="string",
                description=(
                    "脚本工作目录（可选）。"
                    "若脚本内有相对路径引用（如 ./data），必须设置此参数，"
                    "例如 C:\\acm\\AdoDAS2026-main"
                ),
                required=False,
            ),
            ToolParameter(
                name="extra_args",
                type="string",
                description="额外命令行参数，例如 --epochs 50 --lr 0.0001（可选）",
                required=False,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description=f"超时秒数，默认 {DEFAULT_TIMEOUT}",
                required=False,
                default=DEFAULT_TIMEOUT,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        # 兼容旧参数名 script_name
        script_path = (
            parameters.get("script_path")
            or parameters.get("script_name")
            or parameters.get("input")
            or ""
        ).strip()

        if not script_path:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="❌ 缺少参数 script_path",
            )

        working_dir: Optional[str] = parameters.get("working_dir")
        if working_dir:
            working_dir = working_dir.strip() or None

        extra_args_str: str = (parameters.get("extra_args") or "").strip()

        try:
            timeout = int(parameters.get("timeout", self.default_timeout))
        except Exception:
            timeout = self.default_timeout

        cmd = ["python", "-u", script_path]
        if extra_args_str:
            cmd += extra_args_str.split()

        result = _run_subprocess(cmd, timeout=timeout, cwd=working_dir)
        output = result["output"]
        metrics = _parse_metrics(output)

        # ── 构造返回文本 ──────────────────────────────────────
        status_icon = "✅" if result["success"] else "❌"
        lines = [
            f"{status_icon} 训练脚本执行{'成功' if result['success'] else '失败'}",
            f"脚本：{script_path}",
            f"工作目录：{working_dir or '（未指定）'}",
            f"返回码：{result['return_code']}",
        ]

        if result.get("timed_out"):
            lines.append(f"⚠️ 超时（{timeout}秒），进程已终止")

        if metrics:
            lines.append("\n【关键指标】")
            if "f1_cal" in metrics:
                f1 = metrics["f1_cal"]
                flag = "🎉 已达标（≥0.3）" if f1 >= 0.3 else "⚠️ 未达标（<0.3）"
                lines.append(f"  F1_cal = {f1:.4f}  {flag}")
            if "f1_score" in metrics:
                lines.append(f"  F1_score = {metrics['f1_score']:.4f}")
            if "accuracy" in metrics:
                lines.append(f"  Accuracy = {metrics['accuracy']:.4f}")
            if "last_loss" in metrics:
                lines.append(f"  Last Loss = {metrics['last_loss']:.6f}")
            if "last_epoch" in metrics:
                lines.append(f"  Last Epoch = {metrics['last_epoch']}")

        lines.append("\n【训练输出】")
        lines.append(output if output else "[无输出]")

        return_response = ToolResponse.success if result["success"] else ToolResponse.error

        if result["success"]:
            return ToolResponse.success(
                text="\n".join(lines),
                data={
                    "script_path": script_path,
                    "return_code": result["return_code"],
                    "metrics": metrics,
                    "timed_out": result.get("timed_out", False),
                },
            )
        else:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="\n".join(lines),
            )


# ══════════════════════════════════════════════════════════════
# Tool 3：智能读文件
# ══════════════════════════════════════════════════════════════

class SmartReadFileTool(Tool):
    """
    智能读取项目文件或目录结构。

    核心问题修复：
    ─────────────────────────────────────────────────────────────
    原来文件工具有两层截断：
      第一层：工具自身 max_chars=8000（SmartReadFileTool.run）
      第二层：react_agent 的 self.truncator.truncate()，
              其内置阈值通常只有 2000~4000 字符

    两层叠加导致代码文件几乎必然被切断，agent 看到的是不完整的代码，
    无法正确修改或理解逻辑。

    修复策略：
      1. 提升单次读取上限到 20000 字符（覆盖 500-800 行的代码文件）
      2. 新增 start_line / end_line 分段读取，文件很大时 agent 按段读
      3. 返回文件总行数和字符数，让 agent 知道文件规模后决定是否分段
      4. 新增 show_line_numbers 参数，默认开启行号（方便 agent 定位后修改）
      5. 目录列表改为两层展开，agent 一次拿到完整项目结构
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self):
        super().__init__(
            name="read_file",
            description=(
                "读取文件内容或列出目录结构。\n"
                "【读文件】传入 file_path（完整文件路径），可选 start_line/end_line 分段读取\n"
                "【读目录】传入 base_path（目录路径）列出两层目录树\n"
                "【兼容】传入 base_path + file_name 等价于 file_path=base_path/file_name\n"
                "提示：对于大文件，先读取全部获得行数，再用 start_line/end_line 分段读取"
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="文件完整路径（优先）。例如 C:\\acm\\AdoDAS2026-main\\train.py",
                required=False,
            ),
            ToolParameter(
                name="base_path",
                type="string",
                description="目录路径。例如 C:\\acm\\AdoDAS2026-main",
                required=False,
            ),
            ToolParameter(
                name="file_name",
                type="string",
                description="文件名（与 base_path 搭配使用）。例如 train.py",
                required=False,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="从第几行开始读取（从 1 开始，可选）。用于分段读取大文件",
                required=False,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="读到第几行结束（包含，可选）。与 start_line 配合使用",
                required=False,
            ),
            ToolParameter(
                name="show_line_numbers",
                type="boolean",
                description="是否显示行号（默认 true，方便定位代码位置）",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="max_chars",
                type="integer",
                description=f"最大读取字符数，默认 {MAX_FILE_READ_CHARS}，一般不需要手动调整",
                required=False,
                default=MAX_FILE_READ_CHARS,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        file_path: str = (parameters.get("file_path") or "").strip()
        base_path: str = (parameters.get("base_path") or parameters.get("input") or "").strip()
        file_name: str = (parameters.get("file_name") or "").strip()
        start_line: Optional[int] = parameters.get("start_line")
        end_line: Optional[int] = parameters.get("end_line")
        show_line_numbers: bool = parameters.get("show_line_numbers", True)
        max_chars: int = int(parameters.get("max_chars", MAX_FILE_READ_CHARS))

        # ── 路径解析 ─────────────────────────────────────────
        if not file_path:
            if base_path and file_name:
                file_path = os.path.join(base_path, file_name)
            elif base_path:
                # 只给目录 → 列目录结构
                return self._list_dir(base_path)
            else:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=(
                        "❌ 参数不足。请提供以下之一：\n"
                        "  file_path（完整文件路径）\n"
                        "  base_path（目录路径）\n"
                        "  base_path + file_name"
                    ),
                )

        # 如果传入的 file_path 实际上是目录，自动列目录
        if os.path.isdir(file_path):
            return self._list_dir(file_path)

        return self._read_file(
            file_path,
            max_chars=max_chars,
            start_line=int(start_line) if start_line is not None else None,
            end_line=int(end_line) if end_line is not None else None,
            show_line_numbers=bool(show_line_numbers),
        )

    # ──────────────────────────────────────────────────────────
    # 内部：读文件
    # ──────────────────────────────────────────────────────────
    def _read_file(
        self,
        full_path: str,
        max_chars: int,
        start_line: Optional[int],
        end_line: Optional[int],
        show_line_numbers: bool,
    ) -> ToolResponse:

        if not policy.is_allowed_read(full_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"❌ 禁止读取：{full_path}（不在读取白名单）",
            )

        if not os.path.exists(full_path):
            parent = os.path.dirname(full_path)
            hint = ""
            if os.path.isdir(parent):
                try:
                    items = sorted(os.listdir(parent))
                    hint = f"\n\n父目录 {parent} 的内容：\n" + "\n".join(items)
                except Exception:
                    pass
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"❌ 文件不存在：{full_path}{hint}",
            )

        if not os.path.isfile(full_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"❌ 路径不是文件：{full_path}",
            )

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            total_chars = sum(len(l) for l in all_lines)
            file_size_kb = os.path.getsize(full_path) // 1024

            # ── 行范围切片 ───────────────────────────────────
            # start_line/end_line 从 1 开始，转为 0-based index
            s = (start_line - 1) if start_line and start_line >= 1 else 0
            e = end_line if end_line and end_line <= total_lines else total_lines
            s = max(0, min(s, total_lines))
            e = max(s, min(e, total_lines))

            selected_lines = all_lines[s:e]
            is_partial = (s > 0 or e < total_lines)

            # ── 加行号 ───────────────────────────────────────
            if show_line_numbers:
                width = len(str(total_lines))
                numbered = [
                    f"{s + i + 1:>{width}} │ {line}"
                    for i, line in enumerate(selected_lines)
                ]
                content = "".join(numbered)
            else:
                content = "".join(selected_lines)

            # ── 字符数超限处理 ───────────────────────────────
            # 关键：这里的截断只在单次请求的字符确实超过上限时才触发，
            # 而不是像原来那样 8000 字符必截。
            # agent 可以通过 start_line/end_line 分段读取避免截断。
            truncated = False
            if len(content) > max_chars:
                truncated = True
                # 按行截断，避免切断一行中间
                cut_lines = []
                used = 0
                for line in (numbered if show_line_numbers else selected_lines):
                    if used + len(line) > max_chars:
                        break
                    cut_lines.append(line)
                    used += len(line)
                content = "".join(cut_lines)
                actual_end_line = s + len(cut_lines)
            else:
                actual_end_line = e

            # ── 构建返回头部 ─────────────────────────────────
            header_lines = [
                f"📄 文件：{full_path}",
                f"   总行数：{total_lines} 行  |  总字符：{total_chars}  |  大小：{file_size_kb} KB",
            ]

            if is_partial or truncated:
                showing_start = s + 1
                showing_end = actual_end_line
                header_lines.append(
                    f"   当前显示：第 {showing_start} ~ {showing_end} 行"
                )
                remaining = total_lines - actual_end_line
                if remaining > 0:
                    header_lines.append(
                        f"   ⚠️  还有 {remaining} 行未显示。"
                        f"如需继续读取，请使用 start_line={actual_end_line + 1}"
                        f"（可选 end_line={min(actual_end_line + 300, total_lines)}）"
                    )
            else:
                header_lines.append("   ✅ 已显示全部内容")

            header = "\n".join(header_lines)
            separator = "─" * 60
            full_text = f"{header}\n{separator}\n{content}"

            return ToolResponse.success(
                text=full_text,
                data={
                    "full_path": full_path,
                    "total_lines": total_lines,
                    "total_chars": total_chars,
                    "showing_lines": [s + 1, actual_end_line],
                    "truncated": truncated,
                    "has_more": actual_end_line < total_lines,
                },
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"❌ 读取失败：{e}",
            )

    # ──────────────────────────────────────────────────────────
    # 内部：列目录（两层展开）
    # ──────────────────────────────────────────────────────────
    def _list_dir(self, dir_path: str) -> ToolResponse:
        if not policy.is_allowed_read(dir_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"❌ 禁止访问目录：{dir_path}（不在读取白名单）",
            )

        if not os.path.isdir(dir_path):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message=f"❌ 目录不存在：{dir_path}",
            )

        try:
            lines = [f"📁 目录结构：{dir_path}"]
            for item in sorted(os.listdir(dir_path)):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    lines.append(f"├─ 📁 {item}/")
                    try:
                        sub_items = sorted(os.listdir(item_path))
                        for idx, sub in enumerate(sub_items):
                            sub_path = os.path.join(item_path, sub)
                            icon = "📁" if os.path.isdir(sub_path) else "📄"
                            prefix = "│  └─" if idx == len(sub_items) - 1 else "│  ├─"
                            lines.append(f"{prefix} {icon} {sub}")
                    except PermissionError:
                        lines.append("│  └─ [权限不足]")
                else:
                    size = os.path.getsize(item_path)
                    size_str = f"{size // 1024} KB" if size >= 1024 else f"{size} B"
                    lines.append(f"├─ 📄 {item}  ({size_str})")

            return ToolResponse.success(
                text="\n".join(lines),
                data={"dir_path": dir_path},
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"❌ 读取目录失败：{e}",
            )


# ══════════════════════════════════════════════════════════════
# Tool 4：运行 Shell 命令
# ══════════════════════════════════════════════════════════════

class RunShellTool(Tool):
    """
    执行 Shell 命令。

    改进：
    - 支持 working_dir，解决 cd && python 不可靠的问题。
    - 改用 communicate(timeout=) 代替 readline() 轮询。
    - stderr 单独捕获，traceback 不再丢失。
    - 输出截断保留头尾。
    - 失败时返回 ToolResponse.error（原来 return_code!=0 时也 success，
      导致 agent 误以为成功）。
    """

    FORBIDDEN_PATTERNS = ["cat ", "type ", "more ", "less ", "tail ", "head "]

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT):
        super().__init__(
            name="run_shell",
            description=(
                "执行 Shell 命令（如 python train.py、uv add torch、dir 等）。\n"
                "禁止用于查看文件内容（cat/type/more 等），请改用 read_file。\n"
                "若执行 Python 脚本，请优先使用 run_training（含指标解析）。"
            ),
        )
        self.default_timeout = default_timeout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="要执行的 Shell 命令，例如 uv add scikit-learn",
                required=True,
            ),
            ToolParameter(
                name="working_dir",
                type="string",
                description="命令执行目录（可选），例如 C:\\acm\\AdoDAS2026-main",
                required=False,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description=f"超时秒数，默认 {DEFAULT_TIMEOUT}",
                required=False,
                default=DEFAULT_TIMEOUT,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        command = (parameters.get("command") or parameters.get("input") or "").strip()

        if not command:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="❌ 缺少参数 command",
            )

        lower_cmd = command.lower()
        for pat in self.FORBIDDEN_PATTERNS:
            if pat in lower_cmd:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=(
                        f"❌ 禁止用 shell 查看文件内容（检测到：{pat.strip()}）。\n"
                        "请改用 read_file 工具。"
                    ),
                )

        working_dir: Optional[str] = (parameters.get("working_dir") or "").strip() or None

        try:
            timeout = int(parameters.get("timeout", self.default_timeout))
        except Exception:
            timeout = self.default_timeout

        result = _run_subprocess(
            command,
            timeout=timeout,
            cwd=working_dir,
            shell=True,  # shell=True 支持管道、&&、环境变量展开等
        )

        output = result["output"] or "[无输出]"

        lines = [
            f"{'✅' if result['success'] else '❌'} Shell 命令{'成功' if result['success'] else '失败'}",
            f"命令：{command}",
            f"工作目录：{working_dir or '（默认）'}",
            f"返回码：{result['return_code']}",
        ]
        if result.get("timed_out"):
            lines.append(f"⚠️ 超时（{timeout}秒）")
        lines.append("\n【输出】")
        lines.append(output)

        text = "\n".join(lines)

        if result["success"]:
            return ToolResponse.success(
                text=text,
                data={"command": command, "return_code": result["return_code"]},
            )
        else:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=text,
            )


# ══════════════════════════════════════════════════════════════
# 对外导出
# ══════════════════════════════════════════════════════════════

write_file = WriteFileTool()
run_training = RunTrainingTool()
read_file = SmartReadFileTool()
run_shell = RunShellTool()


def register_tools(tool_registry: ToolRegistry) -> ToolRegistry:
    tool_registry.register_tool(write_file)
    tool_registry.register_tool(run_training)
    tool_registry.register_tool(read_file)
    tool_registry.register_tool(run_shell)
    return tool_registry

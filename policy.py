"""
policy.py — 企业级 Agent 访问控制策略引擎

特性：
- 每个工具独立的 allow / deny 规则列表
- 全局 deny 优先级最高（黑名单始终有效）
- glob 通配符匹配（* 单层，** 任意层）
- 规则命中即停止（短路求值）
- 审计日志（可选）
- 默认拒绝（fail-closed）
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml

# ──────────────────────────────────────────────────────────────
# 类型别名
# ──────────────────────────────────────────────────────────────
Action   = Literal["allow", "deny"]
ToolName = Literal["read_file", "write_file", "run_training", "run_shell"]


# ──────────────────────────────────────────────────────────────
# 审计 Logger
# ──────────────────────────────────────────────────────────────
def _build_audit_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("policy.audit")
    if logger.handlers:          # 避免重复添加 handler
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────
# 路径 / 命令规范化
# ──────────────────────────────────────────────────────────────
def _normalize(raw: str, normalize: bool) -> str:
    """
    将路径统一为绝对路径字符串（正斜杠），方便 glob 匹配。
    对 run_shell 的命令字符串不做路径解析，直接小写归一化。
    """
    if not normalize:
        return raw.replace("\\", "/").lower()
    try:
        return str(Path(raw).resolve()).replace("\\", "/")
    except Exception:
        return raw.replace("\\", "/")


def _normalize_pattern(pattern: str) -> str:
    """将 YAML 里写的 Windows 路径模式统一转为正斜杠。"""
    return pattern.replace("\\", "/")


# ──────────────────────────────────────────────────────────────
# Glob 匹配
# ──────────────────────────────────────────────────────────────
def _glob_match(pattern: str, target: str) -> bool:
    """
    支持 ** 的 glob 匹配（大小写不敏感，适配 Windows 路径）。
    ** 展开为 '匹配任意字符包括 /'。
    对于末尾的 foo/**，也要匹配 foo 本身。
    """
    pat = _normalize_pattern(pattern).lower()
    tgt = target.replace("\\", "/").lower()

    # 先特殊处理末尾的 /** 匹配自身目录的语义
    allow_self = False
    if pat.endswith("/**"):
        allow_self = True
        pat = pat[:-3]

    # 将 ** 转为正则中的 .*，* 转为 [^/]*
    # 先把 ** 替换为特殊占位符，再处理单 *
    regex = re.escape(pat)
    regex = regex.replace(r"\*\*", "__DOUBLE_STAR__")
    regex = regex.replace(r"\*",   "[^/]*")
    regex = regex.replace("__DOUBLE_STAR__", ".*")

    if allow_self:
        regex = f"^{regex}(?:/.*)?$"
    else:
        regex = f"^{regex}$"

    return bool(re.match(regex, tgt))


def _matches_any(patterns: list[str], target: str) -> str | None:
    """返回第一个命中的 pattern，未命中返回 None。"""
    for pat in patterns:
        if _glob_match(pat, target):
            return pat
    return None


# ──────────────────────────────────────────────────────────────
# 主策略引擎
# ──────────────────────────────────────────────────────────────
class PathPolicy:
    """
    企业级访问控制策略引擎。

    判断顺序（优先级从高到低）：
      1. global.deny      → DENY（立即拒绝）
      2. tools.<t>.deny   → DENY
      3. tools.<t>.allow  → ALLOW
      4. global.default_action → 兜底
    """

    def __init__(self, config_path: str = os.path.join("config", "policy.yaml")):
        self._cfg:            dict        = {}
        self._global_deny:    list[str]   = []
        self._default_action: Action      = "deny"
        self._normalize:      bool        = True
        self._audit:          bool        = False
        self._audit_logger:   logging.Logger | None = None
        self._tools:          dict[str, dict[str, list[str]]] = {}

        self._load(config_path)

    # ── 加载 ──────────────────────────────────────────────────
    def _load(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"策略文件不存在：{config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self._cfg = cfg
        g = cfg.get("global", {})

        self._default_action = g.get("default_action", "deny")
        self._normalize      = g.get("normalize_paths", True)
        self._audit          = g.get("audit_log", False)
        self._global_deny    = g.get("deny", [])

        if self._audit:
            log_path = g.get("audit_log_path", "logs/policy_audit.log")
            self._audit_logger = _build_audit_logger(log_path)

        # 解析每个工具的规则
        for tool_name, rules in cfg.get("tools", {}).items():
            self._tools[tool_name] = {
                "deny":  rules.get("deny",  []),
                "allow": rules.get("allow", []),
            }

    # ── 公共接口 ──────────────────────────────────────────────
    def check(self, tool: ToolName, target: str) -> bool:
        """
        检查 `tool` 对 `target`（路径或命令）的操作是否被允许。
        返回 True = 允许，False = 拒绝。
        """
        normalize_target = self._normalize if tool != "run_shell" else False
        normalized = _normalize(target, normalize_target)
        action, matched_rule, stage = self._evaluate(tool, normalized)
        allowed = (action == "allow")

        self._audit_log(tool, target, normalized, allowed, stage, matched_rule)
        return allowed

    # 便捷方法（向后兼容 / 语义清晰）
    def is_allowed_read(self, path: str)      -> bool: return self.check("read_file",    path)
    def is_allowed_write(self, path: str)     -> bool: return self.check("write_file",   path)
    def is_allowed_training(self, script: str)-> bool: return self.check("run_training", script)
    def is_allowed_shell(self, command: str)  -> bool: return self.check("run_shell",    command)

    # ── 核心评估逻辑 ──────────────────────────────────────────
    def _evaluate(
        self, tool: str, target: str
    ) -> tuple[Action, str | None, str]:
        """
        返回 (action, 命中的规则, 命中阶段描述)
        """
        # 1. global deny
        hit = _matches_any(self._global_deny, target)
        if hit:
            return "deny", hit, "global.deny"

        tool_rules = self._tools.get(tool, {})

        # 2. tool-level deny
        hit = _matches_any(tool_rules.get("deny", []), target)
        if hit:
            return "deny", hit, f"tools.{tool}.deny"

        # 3. tool-level allow
        hit = _matches_any(tool_rules.get("allow", []), target)
        if hit:
            return "allow", hit, f"tools.{tool}.allow"

        # 4. default
        return self._default_action, None, "default"

    # ── 审计日志 ──────────────────────────────────────────────
    def _audit_log(
        self,
        tool: str,
        raw: str,
        normalized: str,
        allowed: bool,
        stage: str,
        rule: str | None,
    ) -> None:
        if not self._audit or self._audit_logger is None:
            return
        verdict = "ALLOW" if allowed else "DENY"
        rule_str = f'rule="{rule}"' if rule else "rule=<default>"
        self._audit_logger.info(
            f'{verdict} tool={tool} stage={stage} {rule_str} '
            f'raw="{raw}" normalized="{normalized}"'
        )

    # ── 调试用：打印当前策略摘要 ──────────────────────────────
    def summary(self) -> str:
        lines = [
            "═" * 60,
            "  Policy Summary",
            "═" * 60,
            f"  default_action : {self._default_action}",
            f"  normalize_paths: {self._normalize}",
            f"  audit_log      : {self._audit}",
            f"  global.deny    : {len(self._global_deny)} rules",
            "",
        ]
        for tool, rules in self._tools.items():
            lines.append(f"  [{tool}]")
            lines.append(f"    deny : {len(rules['deny'])} rules")
            lines.append(f"    allow: {len(rules['allow'])} rules")
        lines.append("═" * 60)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────────────────────
policy = PathPolicy()

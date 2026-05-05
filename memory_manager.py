"""
memory_manager.py — 分层记忆管理模块

架构：
  long_term_memory.json    ← 精华摘要，token 可控，每次 session 必读
  memory/devlogs/*.json    ← 原始详细日志，按需查阅，不直接注入 prompt

用法：
  from memory_manager import MemoryManager
  mm = MemoryManager()

  # 1. 启动前：生成注入 task 的记忆字符串
  memory_block = mm.build_memory_block()

  # 2. session 结束后：提炼本次经验到长期记忆
  mm.distill(session_devlog_path="memory/devlogs/devlog-xxx.json")
"""

import json
import os
import glob
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────
# 配置常量
# ──────────────────────────────────────────────
LONG_TERM_MEMORY_PATH = "memory/long_term_memory.json"
DEVLOG_DIR            = "memory/devlogs"
RECENT_SESSION_COUNT  = 3    # 注入最近 N 次 session 的 devlog
RECENT_ENTRY_LIMIT    = 10   # 最多取最近 N 条 devlog 条目（防 token 爆炸）
MAX_ENTRY_CHARS       = 300  # 单条 devlog 内容最大字符数（超出截断）


# ──────────────────────────────────────────────
# 长期记忆默认结构
# ──────────────────────────────────────────────
DEFAULT_LONG_TERM = {
    "summary": "",           # 人工/自动维护的精华摘要
    "best_score": None,      # 历史最优 F1_cal
    "best_method": "",       # 最优方法描述
    "best_model_path": "",   # 最优模型路径
    "failed_approaches": [], # 已证明无效的方案（不要重复尝试）
    "key_findings": [],      # 关键发现列表（精简，每条 < 50 字）
    "updated_at": "",
}


class MemoryManager:
    """
    分层记忆管理器

    两层记忆：
      Layer 1 - 长期记忆（long_term_memory.json）
        · 精华摘要，始终注入 prompt
        · 大小受控，不随 session 数量增长

      Layer 2 - 近期 devlog（memory/devlogs/*.json）
        · 只取最近 RECENT_SESSION_COUNT 个文件
        · 只取最近 RECENT_ENTRY_LIMIT 条条目
        · 单条超长时截断
    """

    def __init__(
        self,
        long_term_path: str = LONG_TERM_MEMORY_PATH,
        devlog_dir: str = DEVLOG_DIR,
    ):
        self.long_term_path = long_term_path
        self.devlog_dir = devlog_dir
        os.makedirs(os.path.dirname(long_term_path), exist_ok=True)
        os.makedirs(devlog_dir, exist_ok=True)

    # ══════════════════════════════════════════
    # 公开 API
    # ══════════════════════════════════════════

    def build_memory_block(self) -> str:
        """
        生成注入 task prompt 的记忆块字符串。

        返回示例：
        ──────────────────────────────────────
        【历史经验 - 长期记忆】
        历史最优：XGBoost+SMOTE F1_cal=0.428，模型在 output/a1_best_v2.pt
        ...

        【历史经验 - 近期 Session 记录】
        [progress] A1任务达到F1_cal=0.428，超过目标0.4
        [decision] 最终采用策略：per-class XGBoost...
        ──────────────────────────────────────
        """
        parts = []

        lt = self._load_long_term()
        lt_block = self._format_long_term(lt)
        if lt_block:
            parts.append("【历史经验 - 长期记忆】\n" + lt_block)

        recent_block = self._format_recent_devlogs()
        if recent_block:
            parts.append("【历史经验 - 近期 Session 记录】\n" + recent_block)

        if not parts:
            return ""

        sep = "\n" + "─" * 50 + "\n"
        return sep + sep.join(parts) + sep

    def distill(self, session_devlog_path: Optional[str] = None) -> dict:
        """
        提炼本次 session 经验到长期记忆。

        规则：
          · 如果本次 F1_cal 超过历史最优，更新 best_score / best_method / best_model_path
          · 把本次 decision 类条目中的关键发现追加到 key_findings（去重、限 20 条）
          · 把本次标记为 failed 的方案追加到 failed_approaches（去重、限 15 条）
          · 自动重写 summary

        Args:
            session_devlog_path: 本次 session 的 devlog JSON 路径。
                                 为 None 时自动取 devlog_dir 中最新一个文件。

        Returns:
            更新后的长期记忆 dict
        """
        # 找到本次 devlog
        if session_devlog_path is None:
            files = sorted(glob.glob(os.path.join(self.devlog_dir, "*.json")))
            if not files:
                print("[MemoryManager] 未找到任何 devlog 文件，跳过提炼。")
                return self._load_long_term()
            session_devlog_path = files[-1]

        if not os.path.exists(session_devlog_path):
            print(f"[MemoryManager] devlog 文件不存在: {session_devlog_path}")
            return self._load_long_term()

        with open(session_devlog_path, "r", encoding="utf-8") as f:
            devlog = json.load(f)

        entries = devlog.get("entries", [])
        lt = self._load_long_term()

        # ── 1. 更新最优分数 ──────────────────────
        for entry in entries:
            meta = entry.get("metadata", {})
            f1 = meta.get("f1_cal") or meta.get("f1_cal_xgb")
            if f1 is not None:
                current_best = lt.get("best_score") or 0.0
                if float(f1) > float(current_best):
                    lt["best_score"] = float(f1)
                    lt["best_method"] = meta.get("method", entry.get("content", "")[:80])
                    # 尝试从 content 中提取模型路径
                    content = entry.get("content", "")
                    if "output/" in content:
                        import re
                        match = re.search(r"output/[\w./\-]+\.pt", content)
                        if match:
                            lt["best_model_path"] = match.group(0)

        # ── 2. 追加关键发现 ──────────────────────
        existing_findings = set(lt.get("key_findings", []))
        for entry in entries:
            if entry.get("category") in ("decision", "progress"):
                snippet = entry.get("content", "")[:100].strip()
                if snippet and snippet not in existing_findings:
                    existing_findings.add(snippet)
        lt["key_findings"] = sorted(existing_findings)[-20:]  # 保留最近 20 条

        # ── 3. 追加失败方案 ──────────────────────
        existing_failed = set(lt.get("failed_approaches", []))
        for entry in entries:
            meta = entry.get("metadata", {})
            tags = meta.get("tags", [])
            if "failed" in tags or "无效" in entry.get("content", ""):
                snippet = entry.get("content", "")[:80].strip()
                if snippet:
                    existing_failed.add(snippet)
        lt["failed_approaches"] = sorted(existing_failed)[-15:]

        # ── 4. 重写 summary ──────────────────────
        lt["summary"] = self._build_summary(lt)
        lt["updated_at"] = datetime.now().isoformat()

        self._save_long_term(lt)
        print(f"[MemoryManager] 长期记忆已更新 → {self.long_term_path}")
        return lt

    def update_long_term(self, **kwargs) -> dict:
        """
        手动更新长期记忆字段。

        示例：
            mm.update_long_term(
                best_score=0.47,
                best_method="TCN + attention aggregator",
                best_model_path="output/best_v3.pt"
            )
        """
        lt = self._load_long_term()
        for k, v in kwargs.items():
            if k in lt or k in DEFAULT_LONG_TERM:
                lt[k] = v
        lt["summary"] = self._build_summary(lt)
        lt["updated_at"] = datetime.now().isoformat()
        self._save_long_term(lt)
        return lt

    def get_long_term(self) -> dict:
        """读取长期记忆（只读）"""
        return self._load_long_term()

    # ══════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════

    def _load_long_term(self) -> dict:
        if os.path.exists(self.long_term_path):
            try:
                with open(self.long_term_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 补全缺失字段
                for k, v in DEFAULT_LONG_TERM.items():
                    data.setdefault(k, v)
                return data
            except Exception as e:
                print(f"[MemoryManager] 读取长期记忆失败: {e}，使用默认值。")
        return dict(DEFAULT_LONG_TERM)

    def _save_long_term(self, data: dict):
        with open(self.long_term_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _format_long_term(self, lt: dict) -> str:
        """把长期记忆格式化为可读字符串注入 prompt"""
        lines = []

        if lt.get("best_score") is not None:
            lines.append(f"· 历史最优 F1_cal: {lt['best_score']}")
        if lt.get("best_method"):
            lines.append(f"· 最优方法: {lt['best_method']}")
        if lt.get("best_model_path"):
            lines.append(f"· 最优模型路径: {lt['best_model_path']}")

        if lt.get("failed_approaches"):
            lines.append("· 已证明无效，禁止重试:")
            for fa in lt["failed_approaches"]:
                lines.append(f"  - {fa}")

        if lt.get("key_findings"):
            lines.append("· 关键发现:")
            for kf in lt["key_findings"][-10:]:  # 最多展示 10 条
                lines.append(f"  - {kf}")

        if lt.get("summary"):
            lines.insert(0, lt["summary"])

        return "\n".join(lines)

    def _format_recent_devlogs(self) -> str:
        """格式化最近几次 session 的 devlog 条目"""
        files = sorted(glob.glob(os.path.join(self.devlog_dir, "*.json")))
        recent_files = files[-RECENT_SESSION_COUNT:]

        entries_text = []
        for fp in recent_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session_id = data.get("session_id", os.path.basename(fp))
                entries = data.get("entries", [])
                for entry in entries:
                    content = entry.get("content", "")
                    if len(content) > MAX_ENTRY_CHARS:
                        content = content[:MAX_ENTRY_CHARS] + "..."
                    category = entry.get("category", "info")
                    entries_text.append(f"[{category}] {content}")
            except Exception as e:
                print(f"[MemoryManager] 读取 devlog 失败 ({fp}): {e}")

        # 只保留最近 RECENT_ENTRY_LIMIT 条
        entries_text = entries_text[-RECENT_ENTRY_LIMIT:]
        return "\n".join(entries_text)

    def _build_summary(self, lt: dict) -> str:
        """自动生成 summary 字段"""
        parts = []
        if lt.get("best_score") is not None:
            parts.append(f"历史最优 F1_cal={lt['best_score']}")
        if lt.get("best_method"):
            parts.append(f"方法：{lt['best_method']}")
        if lt.get("best_model_path"):
            parts.append(f"模型：{lt['best_model_path']}")
        if lt.get("failed_approaches"):
            parts.append(f"无效方案（禁止重试）：{'; '.join(lt['failed_approaches'][:5])}")
        return "。".join(parts) + "。" if parts else ""

"""
checkpoint_manager.py — 模型检查点管理核心逻辑

职责：
  - 保存/回滚/列出/删除检查点
  - 维护 checkpoint_registry.json 记录每个检查点的元数据
  - 与 tools.py 解耦，纯文件操作，不依赖 Agent 框架

检查点目录结构：
  C:\\acm\\AdoDAS2026-main\\output\\
    checkpoints\\
      checkpoint_registry.json          ← 检查点注册表
      before_lr_change_20260505_170000.pt
      after_smote_tuning_20260505_183000.pt
      ...
    best.pt                             ← 当前生产模型（Agent 训练的目标）
    a1_best_v2_ensemble.pt              ← 历史最优（可能是之前 session 保存的）
"""

import os
import shutil
import json
import glob
from datetime import datetime
from typing import Optional, List, Dict, Any

# ──────────────────────────────────────────────
# 路径配置（与 tools.py 保持一致）
# ──────────────────────────────────────────────
PROJECT_ROOT      = r"C:\acm\AdoDAS2026-main"
OUTPUT_DIR        = os.path.join(PROJECT_ROOT, "output")
CHECKPOINT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
REGISTRY_FILE     = os.path.join(CHECKPOINT_DIR, "checkpoint_registry.json")

# Agent 训练时默认写入的模型文件名（训练脚本里 torch.save 的路径）
# 如果你的训练脚本保存的不是这个名字，在这里修改
DEFAULT_MODEL_CANDIDATES = [
    "best.pt",
    "a1_best_v2_ensemble.pt",
    "model_best.pt",
    "checkpoint_best.pt",
]

MAX_CHECKPOINTS = 20  # 最多保留检查点数量，超出自动删除最旧的


class CheckpointManager:

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        checkpoint_dir: str = CHECKPOINT_DIR,
        registry_file: str = REGISTRY_FILE,
    ):
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.registry_file = registry_file
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ══════════════════════════════════════════
    # 公开 API
    # ══════════════════════════════════════════

    def save(
        self,
        name: str,
        model_path: Optional[str] = None,
        note: str = "",
        f1_cal: Optional[float] = None,
        trigger: str = "manual",
    ) -> Dict[str, Any]:
        """
        保存当前模型为命名检查点。

        Args:
            name:       检查点名称，如 "before_lr_change"（不含时间戳，自动追加）
            model_path: 源模型文件路径。为 None 时自动搜索 output_dir 下最新的 .pt 文件
            note:       备注，如 "epoch=30, F1=0.441，准备调整学习率"
            f1_cal:     当前 F1 分数（便于后续对比）
            trigger:    触发来源，"manual"=Agent主动调用, "auto"=自动触发

        Returns:
            检查点元数据 dict，失败时包含 "error" 字段
        """
        # 1. 找到源模型文件
        src = self._resolve_model_path(model_path)
        if src is None:
            return {"error": f"未找到模型文件。请确认 output 目录下存在 .pt 文件，或手动指定 model_path。"}
        if not os.path.exists(src):
            return {"error": f"模型文件不存在: {src}"}

        # 2. 生成带时间戳的检查点文件名
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        ckpt_filename = f"{safe_name}_{ts}.pt"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_filename)

        # 3. 复制模型文件
        try:
            shutil.copy2(src, ckpt_path)
        except Exception as e:
            return {"error": f"复制模型文件失败: {e}"}

        # 4. 写入注册表
        entry = {
            "name": name,
            "filename": ckpt_filename,
            "path": ckpt_path,
            "source_path": src,
            "note": note,
            "f1_cal": f1_cal,
            "trigger": trigger,
            "saved_at": datetime.now().isoformat(),
        }
        registry = self._load_registry()
        registry.append(entry)

        # 5. 超出上限则删除最旧的
        if len(registry) > MAX_CHECKPOINTS:
            to_delete = registry[: len(registry) - MAX_CHECKPOINTS]
            for old in to_delete:
                try:
                    if os.path.exists(old["path"]):
                        os.remove(old["path"])
                except Exception:
                    pass
            registry = registry[len(registry) - MAX_CHECKPOINTS :]

        self._save_registry(registry)

        size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        return {
            "ok": True,
            "name": name,
            "filename": ckpt_filename,
            "path": ckpt_path,
            "size_mb": round(size_mb, 2),
            "note": note,
            "f1_cal": f1_cal,
            "saved_at": entry["saved_at"],
        }

    def rollback(
        self,
        name: str,
        target_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        回滚到指定检查点。

        Args:
            name:        检查点名称（模糊匹配，取最近一个）
            target_path: 回滚写入的目标路径。为 None 时写回检查点记录的 source_path

        Returns:
            操作结果 dict
        """
        registry = self._load_registry()

        # 模糊匹配：找所有名字包含 name 的检查点，取最新的
        matches = [e for e in registry if name.lower() in e["name"].lower()]
        if not matches:
            available = [e["name"] for e in registry]
            return {
                "error": f"未找到名称包含 '{name}' 的检查点。",
                "available_checkpoints": available,
            }

        entry = matches[-1]  # 取最新的匹配项
        ckpt_path = entry["path"]

        if not os.path.exists(ckpt_path):
            return {"error": f"检查点文件已丢失: {ckpt_path}"}

        # 确定回滚目标路径
        dst = target_path or entry.get("source_path")
        if not dst:
            dst = self._resolve_model_path(None)
        if not dst:
            return {"error": "无法确定回滚目标路径，请手动指定 target_path"}

        # 备份当前文件（防止误操作）
        if os.path.exists(dst):
            backup_name = f"pre_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            backup_path = os.path.join(self.checkpoint_dir, backup_name)
            try:
                shutil.copy2(dst, backup_path)
            except Exception:
                pass  # 备份失败不阻塞主流程

        # 执行回滚
        try:
            shutil.copy2(ckpt_path, dst)
        except Exception as e:
            return {"error": f"回滚失败: {e}"}

        return {
            "ok": True,
            "rolled_back_to": entry["name"],
            "checkpoint_saved_at": entry["saved_at"],
            "checkpoint_note": entry.get("note", ""),
            "checkpoint_f1_cal": entry.get("f1_cal"),
            "target_path": dst,
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点（按时间从新到旧）"""
        registry = self._load_registry()
        return list(reversed(registry))

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """返回 f1_cal 最高的检查点"""
        registry = self._load_registry()
        with_score = [e for e in registry if e.get("f1_cal") is not None]
        if not with_score:
            return None
        return max(with_score, key=lambda e: e["f1_cal"])

    def delete(self, name: str) -> Dict[str, Any]:
        """删除指定检查点（模糊匹配）"""
        registry = self._load_registry()
        matches = [e for e in registry if name.lower() in e["name"].lower()]
        if not matches:
            return {"error": f"未找到检查点: {name}"}

        deleted = []
        new_registry = []
        for entry in registry:
            if name.lower() in entry["name"].lower():
                try:
                    if os.path.exists(entry["path"]):
                        os.remove(entry["path"])
                    deleted.append(entry["name"])
                except Exception as e:
                    new_registry.append(entry)  # 删除失败则保留
            else:
                new_registry.append(entry)

        self._save_registry(new_registry)
        return {"ok": True, "deleted": deleted}

    # ══════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """解析模型路径：优先用指定路径，否则按优先级搜索"""
        if model_path:
            return model_path

        # 按候选名称搜索
        for candidate in DEFAULT_MODEL_CANDIDATES:
            p = os.path.join(self.output_dir, candidate)
            if os.path.exists(p):
                return p

        # 最后兜底：找 output_dir 下最新修改的 .pt 文件
        pt_files = glob.glob(os.path.join(self.output_dir, "*.pt"))
        # 排除 checkpoints 子目录
        pt_files = [f for f in pt_files if "checkpoints" not in f]
        if pt_files:
            return max(pt_files, key=os.path.getmtime)

        return None

    def _load_registry(self) -> List[Dict]:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_registry(self, registry: List[Dict]):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)


# 全局单例（tools.py 直接 import 使用）
checkpoint_manager = CheckpointManager()

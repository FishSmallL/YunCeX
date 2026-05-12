"""Data-quality tools for competition agents.

These tools intentionally keep cleanlab integration thin: the agent only has to
adapt the current reference model so that it exports out-of-sample predicted
probabilities (and optionally feature embeddings). The first reference model is
typically the official baseline; later rounds should use the current best model.
cleanlab then consumes those universal artifacts, independent of whether the
model is PyTorch, sklearn, XGBoost, HuggingFace, etc.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse
from policy import policy
from project_config import PROJECT_ROOT


def _abs(path: str) -> str:
    return os.path.abspath(path)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_matrix(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return np.load(path)
    return pd.read_csv(path).to_numpy()


def _tool_error(code: str, message: str) -> ToolResponse:
    return ToolResponse.error(code=code, message=message)


class PrepareCleanlabModelSourceTool(Tool):
    """Generate an adaptation plan for extracting cleanlab inputs from the current reference model."""

    def __init__(self):
        super().__init__(
            name="prepare_cleanlab_model_source",
            description=(
                "分析当前参考模型（首轮通常为官方 baseline，后续为当前最优模型）的形态，并生成 cleanlab "
                "适配方案：要求模型导出每个训练样本的 out-of-sample pred_probs，以及可选 feature embeddings。"
            ),
            expandable=False,
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="model_source_path", type="string", description="当前参考模型入口脚本/Notebook/目录路径；首轮通常为官方 baseline，后续为最优模型", required=True),
            ToolParameter(name="model_stage", type="string", description="baseline | best | candidate，用于记录当前 cleanlab 诊断基于哪个模型阶段", required=False, default="baseline"),
            ToolParameter(name="train_data_path", type="string", description="训练集标注文件路径（如 train.csv）", required=True),
            ToolParameter(name="label_column", type="string", description="训练集标签列名", required=True),
            ToolParameter(name="id_column", type="string", description="样本唯一 ID 列名；没有则使用行号", required=False, default=""),
            ToolParameter(name="output_dir", type="string", description="写入适配说明的目录", required=False, default=os.path.join(PROJECT_ROOT, "output", "cleanlab")),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        model_source_path = parameters.get("model_source_path") or parameters.get("baseline_path")
        model_stage = parameters.get("model_stage", "baseline")
        train_data_path = parameters.get("train_data_path")
        label_column = parameters.get("label_column")
        id_column = parameters.get("id_column", "") or "行号"
        output_dir = parameters.get("output_dir") or os.path.join(PROJECT_ROOT, "output", "cleanlab")

        if not model_source_path or not train_data_path or not label_column:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "缺少 model_source_path、train_data_path 或 label_column")
        if not policy.is_allowed_read(model_source_path) or not policy.is_allowed_read(train_data_path):
            return _tool_error(ToolErrorCode.ACCESS_DENIED, "无权读取 model_source_path 或 train_data_path")
        if not policy.is_allowed_write(output_dir):
            return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权写入 output_dir: {output_dir}")

        output_path = _abs(os.path.join(output_dir, "cleanlab_model_source_adapter.md"))
        _ensure_parent(output_path)
        guide = f"""# Cleanlab model-source adapter contract

## Goal
每一轮新训练开始前，复用当前参考模型产生 cleanlab 所需的通用诊断输入。首轮参考模型通常是比赛官方 baseline；后续训练轮次应该改用当前最优模型。
cleanlab 与模型框架解耦，因此不要试图让 cleanlab 理解模型内部实现；只需要让当前参考模型导出统一 artifacts。

## Current model source
- `model_stage`: {model_stage}
- `model_source_path`: {model_source_path}

## Required artifacts
1. `pred_probs`：形状为 `[n_train, n_classes]` 的训练集 out-of-sample 概率。
   - 优先用 K-fold/OOF：每个样本的概率必须来自没有训练过该样本的模型。
   - 若当前参考模型流程只能 train/val split，则先对 train split 做 K-fold 包装。
   - 概率列顺序必须与 `class_names.json` 一致。
2. `labels`：来自 `{train_data_path}` 的 `{label_column}` 列。
3. `sample_id`：来自 `{id_column}`；若没有唯一 ID，使用原始行号。
4. 可选 `features`：模型倒数第二层 embedding、树模型 leaf embedding、或表格特征矩阵。

## How to adapt different model sources
- sklearn/XGBoost/LightGBM：使用 `StratifiedKFold`，每折 `fit()` 后调用 `predict_proba()`。
- PyTorch/Keras/HuggingFace：封装现有 Dataset/DataLoader 和训练循环；每折重新初始化模型，保存验证折 softmax 概率。
- 只有推理脚本的模型：先定位 checkpoint 加载和 `model(input)` 部分，新增对训练集折外 checkpoint 的推理导出。
- Notebook 模型流程：抽出数据读取、模型构建、训练、推理四段到脚本，再套 K-fold。
- 回归/检测/分割等非普通分类任务：先用 cleanlab 支持的对应接口；若暂未适配，则输出当前参考模型的异常分数/embedding 给后续人工规则工具处理。

## Output files expected by `cleanlab_diagnose`
- `pred_probs.npy` 或 `pred_probs.csv`
- optional: `features.npy` 或 `features.csv`
- `class_names.json`

## Agent workflow
1. 读取当前参考模型入口和训练数据 schema。
2. 最小改动：新增一个 `export_cleanlab_artifacts.py`，不要重写官方 baseline 或当前最优模型。
3. 跑通 artifact 导出后调用 `cleanlab_diagnose`。
4. 根据报告再选择数据修复工具；修复后的数据另存为新文件，不覆盖原始数据。
5. 如果执行固定模型的 `data_loop`，每次清洗后都要对新数据再次用同一参考模型导出 OOF `pred_probs`，再次运行 `cleanlab_diagnose`，并调用 `data_quality_loop_policy` 判断是否继续。

## Suggested data_loop policy
- 默认最多 2 轮，不更新模型，只更新数据。
- 仅当 issue_rate 仍 >= 10%、相邻轮次 issue_rate 明显下降、累计改动比例 < 15% 时，才允许第 3 轮。
- 任一停止条件满足即退出：issue_rate <= 1%、剩余疑似问题 < 20、相邻轮次绝对下降 < 0.5 个百分点且相对下降 < 20%、累计改动比例 >= 15%、达到轮数上限。
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(guide)

        return ToolResponse.success(
            text=f"已生成 cleanlab 参考模型适配说明: {output_path}",
            data={"adapter_guide_path": output_path, "required_artifacts": ["pred_probs", "labels", "sample_id"]},
        )


class CleanlabDiagnoseTool(Tool):
    """Run cleanlab on exported reference-model predictions."""

    def __init__(self):
        super().__init__(
            name="cleanlab_diagnose",
            description=(
                "读取当前参考模型导出的 pred_probs/可选 features，并使用 cleanlab 发现疑似标签错误、低质量样本等数据问题。"
            ),
            expandable=False,
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="train_data_path", type="string", description="训练集 CSV 路径", required=True),
            ToolParameter(name="label_column", type="string", description="标签列名", required=True),
            ToolParameter(name="pred_probs_path", type="string", description="OOF pred_probs 的 .npy 或 .csv 路径", required=True),
            ToolParameter(name="output_dir", type="string", description="cleanlab 报告输出目录", required=False, default=os.path.join(PROJECT_ROOT, "output", "cleanlab")),
            ToolParameter(name="id_column", type="string", description="样本 ID 列；不传则用行号", required=False, default=""),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        if importlib.util.find_spec("cleanlab") is None:
            return _tool_error(
                ToolErrorCode.NOT_FOUND,
                "未安装 cleanlab。请先使用允许的包管理命令安装，例如 uv add cleanlab。",
            )

        cleanlab_filter = importlib.import_module("cleanlab.filter")
        cleanlab_rank = importlib.import_module("cleanlab.rank")
        find_label_issues = cleanlab_filter.find_label_issues
        get_label_quality_scores = cleanlab_rank.get_label_quality_scores

        train_data_path = parameters.get("train_data_path")
        label_column = parameters.get("label_column")
        pred_probs_path = parameters.get("pred_probs_path")
        output_dir = parameters.get("output_dir") or os.path.join(PROJECT_ROOT, "output", "cleanlab")
        id_column = parameters.get("id_column", "") or None

        if not train_data_path or not label_column or not pred_probs_path:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "缺少 train_data_path、label_column 或 pred_probs_path")
        for path in (train_data_path, pred_probs_path):
            if not policy.is_allowed_read(path):
                return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权读取: {path}")
        if not policy.is_allowed_write(output_dir):
            return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权写入 output_dir: {output_dir}")

        df = pd.read_csv(train_data_path)
        if label_column not in df.columns:
            return _tool_error(ToolErrorCode.INVALID_PARAM, f"标签列不存在: {label_column}")
        pred_probs = _read_matrix(pred_probs_path)
        if pred_probs.shape[0] != len(df):
            return _tool_error(
                ToolErrorCode.INVALID_PARAM,
                f"pred_probs 行数({pred_probs.shape[0]})与训练集行数({len(df)})不一致；请确认是训练集 OOF 概率。",
            )

        labels_raw = df[label_column].astype("category")
        labels = labels_raw.cat.codes.to_numpy()
        class_names = list(labels_raw.cat.categories.astype(str))
        ranked_indices = find_label_issues(labels=labels, pred_probs=pred_probs, return_indices_ranked_by="self_confidence")
        quality_scores = get_label_quality_scores(labels=labels, pred_probs=pred_probs)
        pred_labels = pred_probs.argmax(axis=1)

        report = pd.DataFrame(
            {
                "row_index": np.arange(len(df)),
                "sample_id": df[id_column].to_numpy() if id_column and id_column in df.columns else np.arange(len(df)),
                "given_label": df[label_column].astype(str).to_numpy(),
                "suggested_label": [class_names[i] for i in pred_labels],
                "label_quality_score": quality_scores,
                "is_label_issue": False,
            }
        )
        report.loc[ranked_indices, "is_label_issue"] = True
        report = report.sort_values(["is_label_issue", "label_quality_score"], ascending=[False, True])

        output_path = _abs(os.path.join(output_dir, "cleanlab_label_issues.csv"))
        summary_path = _abs(os.path.join(output_dir, "cleanlab_summary.json"))
        _ensure_parent(output_path)
        report.to_csv(output_path, index=False)
        summary = {
            "num_rows": int(len(df)),
            "num_classes": int(len(class_names)),
            "class_names": class_names,
            "num_label_issues": int(len(ranked_indices)),
            "issue_rate": float(len(ranked_indices) / max(len(df), 1)),
            "report_path": output_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        text = (
            f"cleanlab 诊断完成：疑似标签问题 {summary['num_label_issues']}/{summary['num_rows']} "
            f"({summary['issue_rate']:.2%})。报告: {output_path}"
        )
        return ToolResponse.success(text=text, data={"summary": summary, "report_path": output_path})


# Backward-compatible class alias for earlier imports. Prefer PrepareCleanlabModelSourceTool.
PrepareCleanlabBaselineTool = PrepareCleanlabModelSourceTool


class DataQualityLoopPolicyTool(Tool):
    """Decide whether another data-only cleanlab loop is worthwhile."""

    def __init__(self):
        super().__init__(
            name="data_quality_loop_policy",
            description=(
                "根据 cleanlab_summary 历史与数据改动比例，判断是否继续同一参考模型下的 data_loop。"
                "默认最多 2 轮，仅在高噪声且问题率持续明显下降时允许第 3 轮。"
            ),
            expandable=False,
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="summary_paths", type="array", description="按迭代顺序排列的 cleanlab_summary.json 路径列表", required=False, default=[]),
            ToolParameter(name="summaries", type="array", description="也可直接传入 cleanlab summary 字典列表", required=False, default=[]),
            ToolParameter(name="iteration", type="integer", description="当前已完成 data_loop 轮数，从 1 开始", required=True),
            ToolParameter(name="max_iterations", type="integer", description="默认最多 2 轮；高噪声且持续改善时才建议 3 轮", required=False, default=2),
            ToolParameter(name="min_issue_rate", type="number", description="问题率低于该值则停止，默认 0.01", required=False, default=0.01),
            ToolParameter(name="min_abs_improvement", type="number", description="相邻两轮 issue_rate 绝对下降小于该值则停止，默认 0.005", required=False, default=0.005),
            ToolParameter(name="min_relative_improvement", type="number", description="相邻两轮 issue_rate 相对下降小于该值则停止，默认 0.20", required=False, default=0.20),
            ToolParameter(name="min_remaining_issues", type="integer", description="剩余疑似问题少于该数量则停止，默认 20", required=False, default=20),
            ToolParameter(name="cumulative_change_rate", type="number", description="本次 data_loop 已累计改动/降权/删除的数据比例", required=False, default=0.0),
            ToolParameter(name="max_cumulative_change_rate", type="number", description="数据累计改动比例超过该值则停止，默认 0.15", required=False, default=0.15),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        summary_paths = parameters.get("summary_paths") or []
        summaries = list(parameters.get("summaries") or [])
        iteration = int(parameters.get("iteration") or 0)
        max_iterations = int(parameters.get("max_iterations") or 2)
        min_issue_rate = float(parameters.get("min_issue_rate", 0.01))
        min_abs_improvement = float(parameters.get("min_abs_improvement", 0.005))
        min_relative_improvement = float(parameters.get("min_relative_improvement", 0.20))
        min_remaining_issues = int(parameters.get("min_remaining_issues", 20))
        cumulative_change_rate = float(parameters.get("cumulative_change_rate", 0.0))
        max_cumulative_change_rate = float(parameters.get("max_cumulative_change_rate", 0.15))

        for path in summary_paths:
            if not policy.is_allowed_read(path):
                return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权读取: {path}")
            with open(path, "r", encoding="utf-8") as f:
                summaries.append(json.load(f))

        if not summaries:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "需要提供 summary_paths 或 summaries")
        if iteration <= 0:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "iteration 必须从 1 开始")

        current = summaries[-1]
        current_issue_rate = float(current.get("issue_rate", 0.0))
        current_issues = int(current.get("num_label_issues", 0))
        reasons: list[str] = []
        should_continue = True

        effective_max = max_iterations
        if max_iterations > 2 and current_issue_rate < 0.10:
            effective_max = 2

        if iteration >= effective_max:
            should_continue = False
            reasons.append(f"已达到 data_loop 上限 {effective_max} 轮")
        if current_issue_rate <= min_issue_rate:
            should_continue = False
            reasons.append(f"当前 issue_rate={current_issue_rate:.4f} 已低于阈值 {min_issue_rate:.4f}")
        if current_issues < min_remaining_issues:
            should_continue = False
            reasons.append(f"剩余疑似问题 {current_issues} 条少于阈值 {min_remaining_issues}")
        if cumulative_change_rate >= max_cumulative_change_rate:
            should_continue = False
            reasons.append(f"累计数据改动比例 {cumulative_change_rate:.2%} 已达到上限 {max_cumulative_change_rate:.2%}")

        improvement = None
        relative_improvement = None
        if len(summaries) >= 2:
            previous_issue_rate = float(summaries[-2].get("issue_rate", 0.0))
            improvement = previous_issue_rate - current_issue_rate
            relative_improvement = improvement / previous_issue_rate if previous_issue_rate > 0 else 0.0
            if improvement < min_abs_improvement and relative_improvement < min_relative_improvement:
                should_continue = False
                reasons.append(
                    "相邻两轮 cleanlab 问题率下降不足："
                    f"abs={improvement:.4f}, rel={relative_improvement:.2%}"
                )

        if should_continue and not reasons:
            reasons.append(
                "仍可继续一轮 data_loop：剩余问题率较高，且未超过轮数/改动比例上限。"
                "下一轮仍应使用同一参考模型重新导出 cleaned 数据上的 OOF pred_probs。"
            )

        recommendation = {
            "should_continue": should_continue,
            "iteration": iteration,
            "effective_max_iterations": effective_max,
            "current_issue_rate": current_issue_rate,
            "current_issues": current_issues,
            "issue_rate_abs_improvement": improvement,
            "issue_rate_relative_improvement": relative_improvement,
            "cumulative_change_rate": cumulative_change_rate,
            "reasons": reasons,
            "default_policy": (
                "data_loop 默认 2 轮；仅当初始/当前 issue_rate >= 10%、每轮问题率明显下降、"
                "且累计改动比例 < 15% 时，才把上限放宽到 3 轮。"
            ),
        }
        text = "继续 data_loop" if should_continue else "停止 data_loop"
        return ToolResponse.success(text=f"{text}: {'; '.join(reasons)}", data=recommendation)


class ApplyCleanlabIssueFixTool(Tool):
    """Apply conservative fixes based on a cleanlab issue report."""

    def __init__(self):
        super().__init__(
            name="apply_cleanlab_issue_fix",
            description=(
                "根据 cleanlab_label_issues.csv 对训练 CSV 做保守修复：删除疑似问题样本、按建议标签重标、或增加样本权重列。"
            ),
            expandable=False,
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="train_data_path", type="string", description="原训练集 CSV 路径", required=True),
            ToolParameter(name="issues_path", type="string", description="cleanlab_diagnose 生成的问题报告 CSV", required=True),
            ToolParameter(name="output_path", type="string", description="修复后 CSV 输出路径；不会覆盖原文件", required=True),
            ToolParameter(name="label_column", type="string", description="训练集标签列", required=True),
            ToolParameter(name="action", type="string", description="drop | relabel | downweight", required=True),
            ToolParameter(name="max_issue_rate", type="number", description="最多自动处理的问题比例，默认 0.15", required=False, default=0.15),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        train_data_path = parameters.get("train_data_path")
        issues_path = parameters.get("issues_path")
        output_path = parameters.get("output_path")
        label_column = parameters.get("label_column")
        action = (parameters.get("action") or "").lower()
        max_issue_rate = float(parameters.get("max_issue_rate", 0.15))

        if not all([train_data_path, issues_path, output_path, label_column, action]):
            return _tool_error(ToolErrorCode.INVALID_PARAM, "缺少必要参数")
        if action not in {"drop", "relabel", "downweight"}:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "action 必须是 drop、relabel 或 downweight")
        for path in (train_data_path, issues_path):
            if not policy.is_allowed_read(path):
                return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权读取: {path}")
        if not policy.is_allowed_write(output_path):
            return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权写入: {output_path}")

        df = pd.read_csv(train_data_path)
        issues = pd.read_csv(issues_path)
        if "row_index" not in issues.columns or "is_label_issue" not in issues.columns:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "issues_path 缺少 row_index 或 is_label_issue 列")
        issue_rows = issues.loc[issues["is_label_issue"].astype(bool), "row_index"].astype(int).tolist()
        limit = int(len(df) * max_issue_rate)
        selected = set(issue_rows[:limit])

        fixed = df.copy()
        if action == "drop":
            fixed = fixed.drop(index=list(selected)).reset_index(drop=True)
        elif action == "relabel":
            if "suggested_label" not in issues.columns:
                return _tool_error(ToolErrorCode.INVALID_PARAM, "relabel 需要 issues_path 包含 suggested_label 列")
            suggestions = issues.set_index("row_index")["suggested_label"].to_dict()
            for row_idx in selected:
                fixed.loc[row_idx, label_column] = suggestions[row_idx]
        else:
            fixed["cleanlab_weight"] = 1.0
            if "label_quality_score" in issues.columns:
                scores = issues.set_index("row_index")["label_quality_score"].to_dict()
                for row_idx in selected:
                    fixed.loc[row_idx, "cleanlab_weight"] = max(float(scores.get(row_idx, 0.0)), 0.05)
            else:
                fixed.loc[list(selected), "cleanlab_weight"] = 0.25

        _ensure_parent(_abs(output_path))
        fixed.to_csv(output_path, index=False)
        return ToolResponse.success(
            text=f"已执行 {action}，处理 {len(selected)} 条问题样本，输出: {output_path}",
            data={"output_path": output_path, "action": action, "num_fixed": len(selected), "num_rows_after": len(fixed)},
        )


class TabularDataRepairTool(Tool):
    """Small deterministic repairs for common tabular data issues."""

    def __init__(self):
        super().__init__(
            name="tabular_data_repair",
            description="对 CSV 执行确定性数据修复：去重、删除缺失行、填充缺失值；始终另存新文件。",
            expandable=False,
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="input_path", type="string", description="输入 CSV 路径", required=True),
            ToolParameter(name="output_path", type="string", description="输出 CSV 路径", required=True),
            ToolParameter(name="action", type="string", description="drop_duplicates | drop_missing | fill_missing", required=True),
            ToolParameter(name="columns", type="array", description="要处理的列；不传表示所有列", required=False, default=[]),
            ToolParameter(name="fill_value", type="string", description="fill_missing 的填充值；为空时数值列用中位数、其他列用众数/UNKNOWN", required=False, default=""),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        input_path = parameters.get("input_path")
        output_path = parameters.get("output_path")
        action = (parameters.get("action") or "").lower()
        columns = parameters.get("columns") or []
        fill_value = parameters.get("fill_value", "")

        if not input_path or not output_path or not action:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "缺少 input_path、output_path 或 action")
        if action not in {"drop_duplicates", "drop_missing", "fill_missing"}:
            return _tool_error(ToolErrorCode.INVALID_PARAM, "action 必须是 drop_duplicates、drop_missing 或 fill_missing")
        if not policy.is_allowed_read(input_path):
            return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权读取: {input_path}")
        if not policy.is_allowed_write(output_path):
            return _tool_error(ToolErrorCode.ACCESS_DENIED, f"无权写入: {output_path}")

        df = pd.read_csv(input_path)
        target_columns = columns or list(df.columns)
        missing = [col for col in target_columns if col not in df.columns]
        if missing:
            return _tool_error(ToolErrorCode.INVALID_PARAM, f"列不存在: {missing}")

        before = len(df)
        repaired = df.copy()
        if action == "drop_duplicates":
            repaired = repaired.drop_duplicates(subset=target_columns).reset_index(drop=True)
        elif action == "drop_missing":
            repaired = repaired.dropna(subset=target_columns).reset_index(drop=True)
        else:
            for col in target_columns:
                if fill_value != "":
                    repaired[col] = repaired[col].fillna(fill_value)
                elif pd.api.types.is_numeric_dtype(repaired[col]):
                    repaired[col] = repaired[col].fillna(repaired[col].median())
                else:
                    mode = repaired[col].mode(dropna=True)
                    repaired[col] = repaired[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")

        _ensure_parent(_abs(output_path))
        repaired.to_csv(output_path, index=False)
        return ToolResponse.success(
            text=f"已执行 {action}：{before} -> {len(repaired)} 行，输出: {output_path}",
            data={"output_path": output_path, "action": action, "rows_before": before, "rows_after": len(repaired)},
        )

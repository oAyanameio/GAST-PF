"""TrajImpute 直接缺失预测评估器（无未来泄漏）。

正式报告只保留普通 minADEK/minFDEK：每个样本在 K 条预测中取最小
误差，再对样本做等权算术平均，不使用模式分数加权。
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def _ade_fde(pred: torch.Tensor, target: torch.Tensor):
    """pred [B, K, T, 2], target [B, T, 2] -> ade [B, K], fde [B, K]"""
    d = torch.norm(pred - target.unsqueeze(1), p=2, dim=-1)  # [B, K, T]
    return d.mean(-1), d[..., -1]


def evaluate_predictions(pred: torch.Tensor, prob: torch.Tensor, target: torch.Tensor):
    """计算普通 minADEK/minFDEK；prob 仅为调用接口兼容而保留。

    Args:
        pred:   [B, K, T, 2]
        prob:   [B, K] 模态分数（普通 min 指标不使用）
        target: [B, T, 2]
    Returns:
        dict: minADEK/minFDEK（每样本一个值的均值前的张量）
    """
    B, K, T, _ = pred.shape
    ade, fde = _ade_fde(pred, target)
    return {
        f"minADE{K}": ade.min(-1).values,
        f"minFDE{K}": fde.min(-1).values,
    }


class DirectEvaluator:
    """按 missing_count 分组累积指标；compute() 输出层级化结果。"""

    def __init__(self):
        # (scene, difficulty, split, missing_count) -> {metric: [values]}
        self.groups = defaultdict(lambda: defaultdict(list))
        # dimension -> value -> {metric: [values]}
        self.dimension_groups = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def update(self, pred, prob, target, scene, difficulty, split, missing_count,
               valid_count=None, anchor_lag=None, forecast_gap=None,
               max_missing_run=None):
        res = evaluate_predictions(pred, prob, target)
        key = (scene, difficulty, split, int(missing_count))
        for name, vals in res.items():
            if name.startswith("_"):
                continue
            self.groups[key][name].extend(vals.tolist())
        dimensions = {"missing_count": int(missing_count)}
        if valid_count is not None:
            dimensions["valid_count"] = int(valid_count)
        if anchor_lag is not None:
            dimensions["anchor_lag"] = int(anchor_lag)
            dimensions["terminal_missing"] = bool(int(anchor_lag) > 0)
        if forecast_gap is not None:
            dimensions["forecast_gap"] = int(forecast_gap)
        if max_missing_run is not None:
            dimensions["max_missing_run"] = int(max_missing_run)
        for dimension, value in dimensions.items():
            value_key = str(value).lower() if isinstance(value, bool) else str(value)
            for name, vals in res.items():
                if name.startswith("_"):
                    continue
                self.dimension_groups[dimension][value_key][name].extend(vals.tolist())

    def compute(self):
        """-> dict[layered results]；每级均含 n 与分 missing_count 明细。"""
        out: dict = {"by_group": {}, "by_dimension": {}, "overall": {}}
        for (scene, diff, split, mc), metrics in sorted(self.groups.items()):
            entry: dict = {"n": len(next(iter(metrics.values())))}
            for name, vals in metrics.items():
                entry[name] = sum(vals) / len(vals)
            out["by_group"][f"{scene}/{diff}/{split}/missing={mc}"] = entry
        for dimension, values in sorted(self.dimension_groups.items()):
            out["by_dimension"][dimension] = {}
            for value, metrics in sorted(values.items(), key=lambda item: item[0]):
                entry = {"n": len(next(iter(metrics.values())))}
                for name, vals in metrics.items():
                    entry[name] = sum(vals) / len(vals)
                out["by_dimension"][dimension][value] = entry
        # Pool per-sample errors directly; do not weight groups or mode scores.
        all_vals = defaultdict(list)
        for metrics in self.groups.values():
            for name, vals in metrics.items():
                all_vals[name].extend(vals)
        for name, vals in all_vals.items():
            out["overall"][name] = sum(vals) / len(vals)
        metric_names = list(all_vals)
        out["overall"]["n"] = (
            len(all_vals[metric_names[0]]) if metric_names else 0
        )
        out["aggregation"] = "unweighted mean over focal samples"
        return out


def save_results(results: dict, meta: dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return out_path

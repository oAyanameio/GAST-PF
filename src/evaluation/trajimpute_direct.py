"""TrajImpute 直接缺失预测评估器（无未来泄漏）。

正式报告只保留普通 minADEK/minFDEK：每个样本在 K 条预测中取最小
误差，再对样本做等权算术平均，不使用模式分数加权。
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import torch


MISSING_HISTORY_FRAME_COUNT = 8
MISSING_SEVERITY_BINS = (
    ("near_complete", "Near-complete", (0, 1, 2)),
    ("moderate", "Moderate", (3, 4)),
    # missing=7 leaves only one observed history frame and is excluded from
    # the severity summary because no observed motion difference is available.
    ("severe", "Severe", (5, 6)),
)
MISSING_SEVERITY_PERCENT_RANGES = {
    "near_complete": "≤25%",
    "moderate": ">25–50%",
    "severe": ">50–75%",
}
MISSING_SEVERITY_LABELS = {
    "near_complete": "Near-complete (≤25%)",
    "moderate": "Moderate (>25–50%)",
    "severe": "Severe (>50–75%)",
}


def missing_percentage(missing_count: int) -> float:
    """Return the missing share of the fixed eight-frame history, in percent."""
    missing_count = int(missing_count)
    if not 0 <= missing_count < MISSING_HISTORY_FRAME_COUNT:
        raise ValueError(
            f"missing_count={missing_count} must be in the supported 0..7 range"
        )
    return 100.0 * missing_count / MISSING_HISTORY_FRAME_COUNT


def classify_missing_severity(missing_count: int) -> str | None:
    """Classify severity; return None for the one-observed-frame extreme."""
    missing_pct = missing_percentage(missing_count)
    if missing_count == MISSING_HISTORY_FRAME_COUNT - 1:
        return None
    if missing_pct <= 25.0:
        return "near_complete"
    if missing_pct <= 50.0:
        return "moderate"
    return "severe"


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
    """Accumulate exact and three-level missingness-stratified metrics."""

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
        """Return exact groups, diagnostic dimensions, and severity summaries."""
        out: dict = {"by_group": {}, "by_dimension": {}, "overall": {}}
        exact_means = {}
        for (scene, diff, split, mc), metrics in sorted(self.groups.items()):
            entry: dict = {"n": len(next(iter(metrics.values())))}
            for name, vals in metrics.items():
                entry[name] = sum(vals) / len(vals)
            group_key = f"{scene}/{diff}/{split}/missing={mc}"
            out["by_group"][group_key] = entry
            exact_means[(scene, diff, split, mc)] = entry

        out["by_severity"] = {}
        severity_keys = sorted(
            {
                (scene, diff, split, severity)
                for scene, diff, split, mc in exact_means
                if (severity := classify_missing_severity(mc)) is not None
            }
        )
        for scene, diff, split, severity in severity_keys:
            levels = next(
                levels
                for key, _, levels in MISSING_SEVERITY_BINS
                if key == severity
            )
            level_entries = [
                exact_means[(scene, diff, split, mc)]
                for mc in levels
                if (scene, diff, split, mc) in exact_means
            ]
            if not level_entries:
                continue
            entry = {
                "n": sum(item["n"] for item in level_entries),
                "label": MISSING_SEVERITY_LABELS[severity],
                "missing_percentage_range":
                    MISSING_SEVERITY_PERCENT_RANGES[severity],
                "missing_levels": [
                    mc for mc in levels
                    if (scene, diff, split, mc) in exact_means
                ],
                "expected_missing_levels": list(levels),
                "aggregation": "equal mean over missing_count means",
            }
            metric_names = set().union(*(item.keys() for item in level_entries))
            metric_names.discard("n")
            metric_names.discard("label")
            metric_names.discard("missing_percentage_range")
            metric_names.discard("missing_levels")
            metric_names.discard("expected_missing_levels")
            metric_names.discard("aggregation")
            for name in sorted(metric_names):
                values = [
                    item[name] for item in level_entries
                    if item.get(name) is not None
                ]
                if values:
                    entry[name] = sum(values) / len(values)

            baseline = exact_means.get((scene, diff, split, 0))
            if baseline is not None:
                for name in tuple(entry):
                    if not name.startswith(("minADE", "minFDE")):
                        continue
                    if baseline.get(name) is not None:
                        entry[f"{name}_delta_vs_missing0"] = (
                            entry[name] - baseline[name]
                        )
                        if baseline[name] != 0:
                            entry[f"{name}_relative_vs_missing0_pct"] = (
                                100.0 * (entry[name] - baseline[name])
                                / baseline[name]
                            )
            out["by_severity"][
                f"{scene}/{diff}/{split}/severity={severity}"
            ] = entry

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

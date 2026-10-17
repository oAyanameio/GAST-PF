"""Stage 0 基线审计：分层维度完整性测试。

Stage 0 要求按 missing_count / valid_count / anchor_lag / forecast_gap /
terminal_missing / 连续缺失长度 六个维度分组统计。前五个评估器已支持，
「连续缺失长度」(max_missing_run) 需要数据层透出并在评估器分层。
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    build_sample, trajimpute_collate_fn, max_missing_run,
)
from src.evaluation.trajimpute_direct import DirectEvaluator  # noqa: E402


def _scene(focal_run: int, neighbor_run: int = 0, actors: int = 2):
    """构造 [A,8,2] 历史：focal(0) 前 focal_run 帧缺失，邻居前 neighbor_run 帧缺失。"""
    hist = torch.zeros(actors, 8, 2, dtype=torch.float64)
    valid = torch.ones(actors, 8, dtype=torch.bool)
    if focal_run:
        valid[0, :focal_run] = False
        hist[0, :focal_run] = float("nan")
    if neighbor_run:
        valid[1, :neighbor_run] = False
        hist[1, :neighbor_run] = float("nan")
    return hist, valid


def _sample(focal_run=0, neighbor_run=0):
    hist, valid = _scene(focal_run, neighbor_run)
    future = torch.zeros(hist.shape[0], 12, 2, dtype=torch.float64)
    sample = build_sample(hist, future, valid, "ETH-M-Easy-test-seq_0", 0)
    # TrajImputeDataset.__getitem__ 在 build_sample 之后补这两个字段
    sample["seq_index"] = 0
    sample["num_actors"] = hist.shape[0]
    return sample


def test_build_sample_emits_focal_max_missing_run():
    sample = _sample(focal_run=3)
    assert sample["max_missing_run"] == 3
    assert sample["max_missing_run"] == max_missing_run(
        sample["x_valid_mask"][0]
    )


def test_build_sample_run_is_focal_not_neighbor():
    sample = _sample(focal_run=0, neighbor_run=5)
    assert sample["max_missing_run"] == 0, "邻居缺失不得计入 focal 分组"


def test_collate_passes_max_missing_run():
    batch = trajimpute_collate_fn([_sample(focal_run=4)])
    assert "max_missing_run" in batch
    assert int(batch["max_missing_run"][0]) == 4


def _pred(n=1):
    return (torch.zeros(n, 20, 12, 2), torch.ones(n, 20), torch.ones(n, 12, 2))


def test_evaluator_stratifies_by_max_missing_run():
    ev = DirectEvaluator()
    pred, prob, target = _pred()
    ev.update(pred, prob, target, "ETH-M", "Mixed", "test", missing_count=3,
              valid_count=5, anchor_lag=0, forecast_gap=1, max_missing_run=3)
    out = ev.compute()
    assert "max_missing_run" in out["by_dimension"]
    assert "3" in out["by_dimension"]["max_missing_run"]
    assert out["by_dimension"]["max_missing_run"]["3"]["n"] == 1


def test_evaluator_keeps_all_six_dimensions_together():
    ev = DirectEvaluator()
    pred, prob, target = _pred()
    ev.update(pred, prob, target, "ETH-M", "Mixed", "test", missing_count=2,
              valid_count=6, anchor_lag=0, forecast_gap=1, max_missing_run=2)
    dims = set(ev.compute()["by_dimension"])
    assert dims == {"missing_count", "valid_count", "anchor_lag",
                    "forecast_gap", "terminal_missing", "max_missing_run"}

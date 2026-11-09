import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from docs.tools import gen_results_figures as result_figures
from docs.tools import visualize_trajgap_scenes as scene_figures


def test_observation_projection_uses_dataset_inverse_transform(
    monkeypatch, tmp_path
):
    Image.new("RGB", (512, 512), color="white").save(tmp_path / "eth.png")
    monkeypatch.setattr(scene_figures, "MAPS_DIR", tmp_path)
    monkeypatch.setattr(scene_figures, "world2pix", lambda scene, xy: np.asarray(xy))

    transformed = []
    original_local2world = scene_figures.local2world

    def record_transform(local, origin, theta):
        world = original_local2world(local, origin, theta)
        transformed.append(world.copy())
        return world

    monkeypatch.setattr(scene_figures, "local2world", record_transform)
    local = torch.zeros((1, 8, 2))
    local[0, 0] = torch.tensor([1.0, 0.0])
    valid = torch.tensor([[True, False, True, False, False, False, False, False]])
    sample = {
        "origin": torch.tensor([[2.0, 3.0]]),
        "theta": torch.tensor([np.pi / 2]),
        "x_positions": local,
        "x_valid_mask": valid,
    }
    result = {
        "gt_world": np.zeros((12, 2)),
        "pred_world": np.zeros((20, 12, 2)),
        "pi": np.full(20, 1 / 20),
        "top1": 0,
        "minfde": 0.0,
    }

    fig, ax = plt.subplots()
    scene_figures.render_one(
        "ETH-M", sample, result, tmp_path / "out.png", ax=ax,
        show_legend=False,
    )

    assert len(transformed) == 1
    np.testing.assert_allclose(transformed[0][0, 0], [2.0, 4.0], atol=1e-6)
    assert len(ax.lines) == 3
    assert all(not line.get_path_effects() for line in ax.lines)
    assert all(line.get_markeredgewidth() == 0 for line in ax.lines[:2])
    plt.close(fig)


def test_missing_observation_frames_break_the_plotted_line():
    points = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    valid = np.array([True, False, True])

    plotted = scene_figures.mask_invalid_observations(points, valid)

    assert np.isnan(plotted[1]).all()
    np.testing.assert_array_equal(plotted[[0, 2]], points[[0, 2]])


def test_benchmark_rows_rank_lower_macro_error_first():
    scenes = result_figures.SCENES
    data = {
        "GAST": {"per": {scene: {"minFDE_K": 0.4} for scene in scenes}},
        "MoFlow": {"per": {scene: {"minFDE_K": 0.6} for scene in scenes}},
    }

    rows = result_figures.benchmark_rows(data, "minFDE_K")

    assert [method for _, method in rows] == ["GAST", "MoFlow"]


def test_best_of_k_index_selects_lowest_final_displacement():
    gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    predictions = np.array([
        [[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.1, 2.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]],
    ])

    assert scene_figures.best_of_k_index(predictions, gt) == 1


def test_visible_sample_selection_uses_upper_quartile_candidate(monkeypatch):
    indices = [0, 1, 2, 3]
    results = {}
    for index, (pred_extent, gt_extent) in enumerate((
        (10.0, 90.0), (90.0, 10.0), (50.0, 50.0), (90.0, 90.0),
    )):
        pred = np.array([
            [0.0, 0.0], [pred_extent / 2, 0.0], [pred_extent, 0.0],
        ])
        gt = np.array([[0.0, 0.0], [gt_extent / 2, 0.0], [gt_extent, 0.0]])
        results[index] = {
            "pred_world": pred[None], "gt_world": gt,
            "obs_world": np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0]]),
            "obs_valid": np.array([True, True, True]),
        }

    monkeypatch.setattr(
        scene_figures, "world2pix", lambda scene, xy: np.asarray(xy)
    )
    (selected, pred_span, pred_arc, gt_span, gt_arc, obs_span, obs_arc,
     obs_run, eligible) = scene_figures.pick_visible_sample(
        results, indices, "ETH-M", min_span_px=28, min_arc_px=40
    )

    assert selected == 3
    assert pred_span == gt_span == 90.0
    assert pred_arc == gt_arc == 90.0
    assert obs_span == obs_arc == 40.0
    assert obs_run == 3
    assert eligible == 2


def test_moflow_grid_uses_three_missingness_rows_and_shared_legend():
    assert [label for _, label in scene_figures.MOFLOW_LAYERS] == [
        "0 missing", "1-4 missing", "5+ missing",
    ]
    assert [handle.get_label() for handle in scene_figures.moflow_legend_handles()] == [
        "Past trajectory", "Ground truth",
        "Best-of-20 prediction (oracle minFDE)",
    ]

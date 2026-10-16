import json
import sys
from pathlib import Path

import torch
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "训练与评估"
sys.path.insert(0, str(SCRIPT_DIR))

from run_trajimpute_experiments import latest_valid_resume_checkpoint
from run_trajimpute_experiments import rebase_checkpoint_callback_paths


def _signature():
    return [{
        "weight_decay": 0.01,
        "parameters": [{"name": "net.weight", "shape": [1]}],
    }]


def _contract():
    return {"contract_version": 1}


def _write_checkpoint(path, *, epoch, global_step, state_dict=None,
                      optimizer_states=None, contract=None, signature=None):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": state_dict or {"net.weight": torch.ones(1)},
        "optimizer_states": optimizer_states or [{
            "state": {
                0: {"step": torch.tensor(1.0), "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)},
            },
            "param_groups": [{"params": [0], "weight_decay": 0.01}],
        }],
        "trajgap_run_contract": contract or _contract(),
        "trajgap_optimizer_signature": signature or _signature(),
    }, path)


def test_resume_selects_furthest_versioned_last_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    older = checkpoint_dir / "last.ckpt"
    newer = checkpoint_dir / "last-v1.ckpt"
    _write_checkpoint(older, epoch=69, global_step=63770)
    _write_checkpoint(newer, epoch=76, global_step=70147)

    assert latest_valid_resume_checkpoint(checkpoint_dir) == newer


def test_resume_skips_invalid_newest_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    valid = checkpoint_dir / "last.ckpt"
    corrupt = checkpoint_dir / "last-v1.ckpt"
    _write_checkpoint(valid, epoch=20, global_step=1000)
    corrupt.write_bytes(b"partial checkpoint")

    assert latest_valid_resume_checkpoint(checkpoint_dir) == valid


def test_resume_rejects_unmigrated_optimizer_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    legacy = checkpoint_dir / "last.ckpt"
    _write_checkpoint(
        legacy,
        epoch=20,
        global_step=1000,
        state_dict={"net.unitraj_encoder.weight": torch.ones(1)},
    )

    with pytest.raises(RuntimeError, match="migrate model and optimizer state"):
        latest_valid_resume_checkpoint(checkpoint_dir)


def test_resume_returns_none_without_checkpoint(tmp_path):
    assert latest_valid_resume_checkpoint(tmp_path) is None


def test_resume_rebases_stale_callback_paths_without_mutating_source(tmp_path):
    checkpoint_dir = tmp_path / "lane_b" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "epoch=70.ckpt"
    best.write_bytes(b"best model")
    source = checkpoint_dir / "last.ckpt"
    torch.save({
        "epoch": 70,
        "global_step": 62196,
        "state_dict": {"net.weight": torch.ones(1)},
        "optimizer_states": [{"state": {}, "param_groups": [{}]}],
        "callbacks": {
            "ModelCheckpoint": {
                "best_model_path": str(tmp_path / "lane_c" / "checkpoints" / best.name),
                "best_k_models": {
                    str(tmp_path / "lane_c" / "checkpoints" / best.name): torch.tensor(0.5),
                },
            }
        },
    }, source)

    rebased = rebase_checkpoint_callback_paths(source, checkpoint_dir)
    payload = torch.load(rebased, map_location="cpu", weights_only=False)
    callback = payload["callbacks"]["ModelCheckpoint"]
    assert rebased != source
    assert callback["best_model_path"] == str(best.resolve())
    assert list(callback["best_k_models"]) == [str(best.resolve())]
    original = torch.load(source, map_location="cpu", weights_only=False)
    assert "lane_c" in original["callbacks"]["ModelCheckpoint"]["best_model_path"]


def test_checkpoint_migrator_refuses_already_migrated_checkpoint(tmp_path):
    from migrate_utenc_ckpt import migrate

    checkpoint = tmp_path / "already_migrated.ckpt"
    torch.save({
        "state_dict": {"net.gast_encoder.weight": torch.ones(1)},
        "optimizer_states": [],
    }, checkpoint)

    with pytest.raises(RuntimeError, match="already migrated"):
        migrate(checkpoint)


def test_selection_monitor_is_final_output_head_metric():
    from run_trajimpute_experiments import checkpoint_monitor

    assert checkpoint_monitor(20) == "val_new_minFDE20"


def test_resume_manifest_rejects_historical_revision_or_policy(tmp_path):
    from run_trajimpute_experiments import validate_resume_manifest

    expected = {
        "resume_contract": {
            "contract_version": 1,
            "git_revision": "new-revision",
            "optimizer_policy": "adamw-v2",
            "monitor": "val_new_minFDE20",
        }
    }
    old = {
        "resume_contract": dict(
            expected["resume_contract"],
            git_revision="old-revision",
            optimizer_policy="adamw-v1",
        )
    }

    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        validate_resume_manifest(old, expected)


def test_optimizer_preflight_rejects_mismatched_moment_shape():
    from run_trajimpute_experiments import validate_optimizer_state_shapes

    optimizer_state = {
        "state": {
            0: {"step": torch.tensor(3.0), "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)},
            1: {"step": torch.tensor(3.0), "exp_avg": torch.ones(3), "exp_avg_sq": torch.ones(3)},
        },
        "param_groups": [
            {"params": [0], "weight_decay": 0.01},
            {"params": [1], "weight_decay": 0.0},
        ],
    }
    signature = [
        {"weight_decay": 0.01, "parameters": [{"name": "linear.weight", "shape": [2]}]},
        {"weight_decay": 0.0, "parameters": [{"name": "mixer.weight", "shape": [4]}]},
    ]

    with pytest.raises(RuntimeError, match="optimizer state shape mismatch"):
        validate_optimizer_state_shapes(optimizer_state, signature)


def test_callback_rebase_updates_dirpath_even_when_old_path_still_exists(tmp_path):
    checkpoint_dir = tmp_path / "lane_b" / "checkpoints"
    old_dir = tmp_path / "lane_c" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    best = checkpoint_dir / "epoch=70.ckpt"
    best.write_bytes(b"new lane checkpoint")
    (old_dir / best.name).write_bytes(b"stale old lane checkpoint")
    source = checkpoint_dir / "last.ckpt"
    torch.save({
        "epoch": 70,
        "global_step": 62196,
        "state_dict": {"net.weight": torch.ones(1)},
        "optimizer_states": [{"state": {}, "param_groups": [{}]}],
        "callbacks": {
            "ModelCheckpoint": {
                "dirpath": str(old_dir),
                "best_model_path": str(old_dir / best.name),
                "best_k_models": {str(old_dir / best.name): torch.tensor(0.5)},
            }
        },
    }, source)

    rebased = rebase_checkpoint_callback_paths(source, checkpoint_dir)
    payload = torch.load(rebased, map_location="cpu", weights_only=False)
    callback = payload["callbacks"]["ModelCheckpoint"]
    assert callback["dirpath"] == str(checkpoint_dir.resolve())
    assert callback["best_model_path"] == str(best.resolve())
    assert list(callback["best_k_models"]) == [str(best.resolve())]


def test_output_directory_lock_rejects_concurrent_runner(tmp_path):
    from run_trajimpute_experiments import output_directory_lock

    run_dir = tmp_path / "same-run"
    with output_directory_lock(run_dir):
        with pytest.raises(RuntimeError, match="already locked"):
            with output_directory_lock(run_dir):
                pass


def test_resume_preflight_rejects_checkpoint_from_other_contract(tmp_path):
    from run_trajimpute_experiments import latest_valid_resume_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _write_checkpoint(checkpoint_dir / "last.ckpt", epoch=10, global_step=100,
                      contract={"contract_version": 1, "monitor": "val_minFDE20"})
    expected_manifest = {"resume_contract": {"contract_version": 1,
                                             "monitor": "val_new_minFDE20"}}

    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        latest_valid_resume_checkpoint(checkpoint_dir, expected_manifest, _signature())


def test_resume_preflight_accepts_matching_contract(tmp_path):
    from run_trajimpute_experiments import latest_valid_resume_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "last.ckpt"
    _write_checkpoint(checkpoint, epoch=10, global_step=100)
    expected_manifest = {"resume_contract": _contract()}

    assert latest_valid_resume_checkpoint(
        checkpoint_dir, expected_manifest, _signature()
    ) == checkpoint


def test_resume_preflight_rejects_missing_optimizer_signature(tmp_path):
    from run_trajimpute_experiments import latest_valid_resume_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "last.ckpt"
    torch.save({
        "epoch": 10,
        "global_step": 100,
        "state_dict": {"net.weight": torch.ones(1)},
        "optimizer_states": [{"state": {}, "param_groups": [{"params": [0],
                                                            "weight_decay": 0.01}]}],
        "trajgap_run_contract": _contract(),
    }, checkpoint)

    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        latest_valid_resume_checkpoint(
            checkpoint_dir, {"resume_contract": _contract()}, _signature()
        )


def _write_eval_result(root, scene, difficulty, variant, seed, n, split="test"):
    result_dir = Path(root) / f"{scene}_{difficulty}_test_{variant}_seed{seed}_ckpt"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "results.json").write_text(json.dumps({
        "meta": {"split": split, "difficulty": difficulty},
        "results": {"overall": {"n": n, "minFDE_K": 0.8}},
    }))
    return result_dir


def test_eval_output_reuses_valid_existing_result(tmp_path):
    from run_trajimpute_experiments import prepare_eval_output

    root = tmp_path / "eval" / "Mixed"
    result_dir = _write_eval_result(root, "ETH-M", "Mixed", "GAST-PF", 2024, 1629)

    payload = prepare_eval_output(root, "ETH-M", "Mixed", "GAST-PF", 2024)
    assert payload is not None
    assert payload["results"]["overall"]["n"] == 1629
    assert (result_dir / "results.json").is_file()


def test_eval_output_clears_zero_sample_partial_result(tmp_path):
    from run_trajimpute_experiments import prepare_eval_output

    root = tmp_path / "eval" / "Mixed"
    result_dir = _write_eval_result(root, "ETH-M", "Mixed", "GAST-PF", 2024, 0)

    assert prepare_eval_output(root, "ETH-M", "Mixed", "GAST-PF", 2024) is None
    assert not result_dir.exists()


def test_eval_output_leaves_other_scene_results_untouched(tmp_path):
    from run_trajimpute_experiments import prepare_eval_output

    root = tmp_path / "eval" / "Mixed"
    other = _write_eval_result(root, "HOTEL-M", "Mixed", "GAST-PF", 2024, 9477)
    stale = root / "ETH-M_Mixed_test_GAST-PF_seed2024_ckpt"
    stale.mkdir(parents=True)
    (stale / "partial.tmp").write_text("half-written")

    assert prepare_eval_output(root, "ETH-M", "Mixed", "GAST-PF", 2024) is None
    assert not stale.exists()
    assert (other / "results.json").is_file()


def test_folds_map_to_scene_names():
    from run_trajimpute_experiments import resolve_scenes

    assert resolve_scenes(None, ["ETH"]) == ["ETH-M"]
    assert resolve_scenes(None, ["ETH", "ZARA2"]) == ["ETH-M", "ZARA2-M"]
    assert resolve_scenes(None, None) == ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
    assert resolve_scenes(["HOTEL-M"], None) == ["HOTEL-M"]


def test_folds_and_scenes_are_mutually_exclusive():
    from run_trajimpute_experiments import resolve_scenes

    with pytest.raises(SystemExit):
        resolve_scenes(["ETH-M"], ["ETH"])

"""TrajGap 缺失数据重训入口（统一 Mixed direct 协议）。

协议（任务书 2026-09-07 + 方案 §1.4）：
- 默认从 `GAST-PF + mixed-direct` 开始；
- `mixed-direct` 将官方 Easy/Hard 的同一 split 合并为单一 Mixed 数据集；
- 每个场景从头训练，不使用零样本 checkpoint；
- 训练输出、验证选点和测试评估统一使用 K=20；
- 每个实验写入独立 manifest，拒绝把不同数据源混为同一协议；
  - variant -> 模型开关固定映射；
- 失败时保留完整 traceback 到 runner.log。

用法：
  PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/训练与评估/run_trajimpute_experiments.py \
      --protocol mixed-direct --variant GAST-PF \
      --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
      --seed 2024 --gpu 3 --output-root outputs/trajgap_bench
"""

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
CONFIG_NAME = "config_missing_aware_trajimpute"
EVAl_SCRIPT = REPO / "scripts/结果分析/evaluate_trajimpute_direct.py"

SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
DIRECT_NUM_MODES = 20
DEFAULT_PROTOCOL = "mixed-direct"
DEFAULT_VARIANT = "GAST-PF"
OPTIMIZER_POLICY = "adamw_complete_module_local_v1"

# Variants: M0 reference；GAST-PF = gap-aware spatio-temporal encoder
# replacement. The decoder, losses, and TrajGap protocol stay unchanged.
VARIANTS = {
    "M0": {"encoder_type": "m0"},
    "GAST-PF": {
        "encoder_type": "gast",
        "gast_depth": 4,
        "gast_gsm_operator": "max",
    },
    # Legacy experiment label; keep it loadable without making it the current
    # public name.
    "UT-ENC": {
        "encoder_type": "utenc",
        "utenc_depth": 4,
        "utenc_gsm_operator": "max",
    },
}

PROTOCOLS = {
    "mixed-direct": {
        "dataset": "trajgap",
        "difficulty": "Mixed",
        "train_difficulties": ["Mixed"],
        "val_difficulties": ["Mixed"],
        "test_difficulties": ["Mixed"],
        "train_sampling_scheme": "natural",
        "zero_missing_only": False,
        "suffix": "_trajgap",
    },
}

RESUME_CONTRACT_VERSION = 1
RESUME_CONTRACT_FIELDS = (
    "protocol", "dataset", "data_root", "files", "scene", "model_version",
    "encoder_type", "encoder_config", "model_config", "seed", "batch_size",
    "epochs", "precision", "lr", "weight_decay", "optimizer_policy",
    "train_sampling_scheme", "train_sampling_seed", "monitor", "K",
    "backbone", "initialization_checkpoint", "git_revision",
    "runner_contract_version", "optimizer_signature",
)


def resolve_scenes(scenes, folds):
    """Map the legacy --folds spelling onto scene names; the two are exclusive."""
    if folds and scenes:
        raise SystemExit("--folds 与 --scenes 互斥；请只用其中一种指定场景")
    if folds:
        return [f"{fold}-M" for fold in folds]
    if scenes:
        return list(scenes)
    return list(SCENES)


def checkpoint_monitor(num_modes):
    if num_modes != DIRECT_NUM_MODES:
        raise ValueError(f"TrajGap-Bench requires K={DIRECT_NUM_MODES}")
    return f"val_new_minFDE{num_modes}"


def make_resume_contract(manifest):
    return {key: manifest.get(key) for key in RESUME_CONTRACT_FIELDS} | {
        "contract_version": RESUME_CONTRACT_VERSION,
    }


def validate_resume_manifest(existing, expected):
    old_contract = existing.get("resume_contract")
    expected_contract = expected.get("resume_contract")
    if not isinstance(old_contract, dict) or not isinstance(expected_contract, dict):
        raise RuntimeError("resume contract mismatch: legacy manifest has no versioned contract")
    mismatches = [
        key for key in sorted(set(old_contract) | set(expected_contract))
        if old_contract.get(key) != expected_contract.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "resume contract mismatch for " + ", ".join(mismatches)
            + "; use a new output root instead of mixing experiment artifacts"
        )


def validate_optimizer_state_shapes(optimizer_state, signature):
    """Check stored optimizer group order, decay policy and moment tensor shapes."""
    groups = optimizer_state.get("param_groups", [])
    states = optimizer_state.get("state", {})
    if len(groups) != len(signature):
        raise RuntimeError("optimizer state shape mismatch: parameter-group count differs")
    for group_index, (group, expected) in enumerate(zip(groups, signature)):
        params = group.get("params", [])
        entries = expected.get("parameters", [])
        if len(params) != len(entries):
            raise RuntimeError(
                f"optimizer state shape mismatch: group {group_index} parameter count differs"
            )
        if abs(float(group.get("weight_decay", 0.0)) - float(expected["weight_decay"])) > 1e-12:
            raise RuntimeError(
                f"optimizer state shape mismatch: group {group_index} weight_decay differs"
            )
        for param_id, parameter in zip(params, entries):
            state = states.get(param_id, states.get(str(param_id)))
            if state is None:
                continue
            expected_shape = tuple(parameter["shape"])
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if value is not None and tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        "optimizer state shape mismatch: "
                        f"{parameter['name']} {key} {tuple(value.shape)} != {expected_shape}"
                    )


def expected_optimizer_signature(manifest):
    """Instantiate the configured model and return actual optimizer parameter order."""
    from src.model.forecasting_module import ForecastingLightningModule

    model_config = dict(manifest["model_config"])
    model_config["type"] = "TrajectoryForecaster"
    module = ForecastingLightningModule(
        model=model_config,
        lr=manifest["lr"],
        weight_decay=manifest["weight_decay"],
        epochs=manifest["epochs"],
        warmup_epochs=5,
    )
    optimizers, _ = module.configure_optimizers()
    optimizer = optimizers[0]
    name_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    signature = []
    for group in optimizer.param_groups:
        signature.append({
            "weight_decay": float(group["weight_decay"]),
            "parameters": [
                {"name": name_by_id[id(parameter)], "shape": list(parameter.shape)}
                for parameter in group["params"]
            ],
        })
    return signature


def validate_resume_checkpoint(checkpoint_path, expected_manifest, expected_signature):
    import torch

    with Path(checkpoint_path).open("rb") as handle:
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        except TypeError:
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu")
    if payload.get("trajgap_run_contract") != expected_manifest.get("resume_contract"):
        raise RuntimeError("resume contract mismatch in checkpoint metadata")
    if payload.get("trajgap_optimizer_signature") != expected_signature:
        raise RuntimeError("resume contract mismatch: optimizer parameter names/order/shapes differ")
    state_dict = payload.get("state_dict", {})
    for group in expected_signature:
        for parameter in group["parameters"]:
            key = parameter["name"]
            tensor = state_dict.get(key)
            if tensor is None or list(tensor.shape) != parameter["shape"]:
                raise RuntimeError(f"resume contract mismatch: model parameter shape differs for {key}")
    optimizer_states = payload.get("optimizer_states") or []
    if len(optimizer_states) != 1:
        raise RuntimeError("resume contract mismatch: expected exactly one optimizer state")
    validate_optimizer_state_shapes(optimizer_states[0], expected_signature)
    return payload


@contextmanager
def output_directory_lock(run_dir):
    """Acquire an exclusive non-blocking lock for one experiment output directory."""
    run_dir = Path(run_dir)
    lock_path = run_dir.parent / f".{run_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"output directory already locked: {run_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _model_config(args):
    config = {
        "embed_dim": 128, "future_steps": 12, "num_heads": 8,
        "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
        "num_actor_types": 1, "num_modes": args.K, "dt": 0.4,
        "obs_len": 8, "bimamba": args.bimamba,
        "encoder_type": "gast", "gast_depth": 4,
        "gast_gsm_operator": "max", "utenc_depth": 4,
        "utenc_gsm_operator": "max",
    }
    config.update(VARIANTS[args.variant])
    return config


def get_git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def env_info():
    import torch
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def validate_paired_init(checkpoint, seed, num_modes):
    """Validate the stage-0 initialization artifact against the run contract."""
    import torch

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"paired initialization checkpoint not found: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    meta = payload.get("meta", {})
    if meta.get("initialization_only") is not True:
        raise ValueError("paired init must be a stage-0 initialization-only checkpoint")
    if int(meta.get("seed", -1)) != seed:
        raise ValueError("paired init seed must match --seed")
    if int(meta.get("num_modes", -1)) != num_modes:
        raise ValueError("paired init K must match --K")


def latest_valid_resume_checkpoint(checkpoint_dir, expected_manifest=None,
                                   expected_signature=None):
    """Return the furthest checkpoint that passes the full resume preflight."""
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    candidates = sorted(
        checkpoint_dir.glob("last*.ckpt"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    valid = []
    failures = []
    for path in candidates:
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            state_dict = payload.get("state_dict")
            optimizer_states = payload.get("optimizer_states")
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError("missing/non-dict state_dict")
            if not isinstance(optimizer_states, list) or not optimizer_states:
                raise ValueError("missing optimizer state; refusing weights-only resume")
            if (expected_manifest is None) != (expected_signature is None):
                raise ValueError("resume preflight requires both manifest and optimizer signature")
            if expected_manifest is not None:
                validate_resume_checkpoint(path, expected_manifest, expected_signature)
            elif any("unitraj_encoder." in key for key in state_dict):
                raise ValueError(
                    "legacy unitraj_encoder checkpoint; migrate model and optimizer "
                    "state before resuming"
                )
            epoch = int(payload.get("epoch", -1))
            global_step = int(payload.get("global_step", -1))
            if epoch < 0 or global_step < 0:
                raise ValueError("missing epoch/global_step metadata")
            valid.append((global_step, epoch, path.stat().st_mtime_ns,
                          path.name, path))
            del payload
        except Exception as error:
            failures.append(f"{path.name}: {error}")

    if not valid and failures:
        raise RuntimeError(
            f"No valid resumable checkpoint in {checkpoint_dir}: "
            + "; ".join(failures)
        )
    if not valid:
        return None
    return max(valid)[-1]


def rebase_checkpoint_callback_paths(checkpoint_path, checkpoint_dir):
    """Rebase stale ModelCheckpoint paths in a copy, never the source checkpoint."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = Path(checkpoint_dir)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    changed = 0
    checkpoint_dir = checkpoint_dir.resolve()

    def rebase_checkpoint(value):
        nonlocal changed
        if not isinstance(value, str) or not value:
            return ""
        current_path = checkpoint_dir / Path(value).name
        if current_path.is_file():
            resolved = str(current_path.resolve())
            if resolved != value:
                changed += 1
            return resolved
        changed += 1
        return ""

    for callback_state in payload.get("callbacks", {}).values():
        if not isinstance(callback_state, dict):
            continue
        if "dirpath" in callback_state:
            new_dir = str(checkpoint_dir)
            if callback_state["dirpath"] != new_dir:
                callback_state["dirpath"] = new_dir
                changed += 1
        for key in ("best_model_path", "last_model_path", "kth_best_model_path"):
            if key in callback_state:
                callback_state[key] = rebase_checkpoint(callback_state[key])
        best_k_models = callback_state.get("best_k_models")
        if isinstance(best_k_models, dict):
            rebased_models = {}
            for path, score in best_k_models.items():
                rebased = rebase_checkpoint(path)
                if rebased:
                    rebased_models[rebased] = score
            callback_state["best_k_models"] = rebased_models

    if not changed:
        return checkpoint_path

    step = int(payload.get("global_step", 0))
    rebased_path = checkpoint_dir / f"last-resume-{step}.ckpt"
    temporary_path = rebased_path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(rebased_path)
    print(
        f"[resume-rebase] {checkpoint_path.name} -> {rebased_path.name} "
        f"({changed} callback paths)",
        flush=True,
    )
    return rebased_path


def eval_result_dir(output_root, scene, difficulty, variant, seed):
    return Path(output_root) / f"{scene}_{difficulty}_test_{variant}_seed{seed}_ckpt"


def prepare_eval_output(output_root, scene, difficulty, variant, seed):
    """Idempotent evaluation output: reuse a valid result, clear only invalid partials.

    Returns the existing valid payload when the cell is already evaluated, else
    removes that cell's incomplete directory and returns None so the caller can
    re-run the evaluator without tripping its overwrite guard.
    """
    result_dir = eval_result_dir(output_root, scene, difficulty, variant, seed)
    result_path = result_dir / "results.json"
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text())
        except Exception:
            payload = None
        if payload is not None and payload.get("results", {}).get("overall", {}).get("n"):
            return payload
    if result_dir.exists():
        shutil.rmtree(result_dir)
    return None


def select_best_checkpoint(train_dir, monitor):
    """按验证集 monitor 选择 checkpoint，禁止用最后 epoch 代替选点。"""
    train_dir = Path(train_dir)
    metrics_files = sorted((train_dir / "logs").glob("version_*/metrics.csv"))
    if not metrics_files:
        raise RuntimeError(f"未找到 metrics.csv: {train_dir / 'logs'}")

    best = None
    for metrics_file in metrics_files:
        with metrics_file.open() as handle:
            for row in csv.DictReader(handle):
                value = row.get(monitor)
                epoch = row.get("epoch")
                if value in (None, "") or epoch in (None, ""):
                    continue
                candidate = (int(float(epoch)), float(value))
                if best is None or candidate[1] < best[1]:
                    best = candidate
    if best is None:
        raise RuntimeError(f"metrics.csv 没有指标 {monitor}: {train_dir}")

    best_epoch, best_value = best
    exact = train_dir / "checkpoints" / f"epoch={best_epoch}.ckpt"
    candidates = [exact] if exact.exists() else sorted(
        (train_dir / "checkpoints").glob(f"epoch={best_epoch}*.ckpt")
    )
    if not candidates:
        raise RuntimeError(
            f"最佳 epoch={best_epoch} 的 checkpoint 不存在: "
            f"{train_dir / 'checkpoints'}"
        )
    return best_epoch, best_value, str(candidates[0])


def build_manifest(args, protocol_cfg):
    data_root = Path(args.data_root)
    train_difficulties = protocol_cfg.get("train_difficulties", [protocol_cfg["difficulty"]])
    val_difficulties = protocol_cfg.get("val_difficulties", [protocol_cfg["difficulty"]])
    test_difficulties = protocol_cfg.get("test_difficulties", [protocol_cfg["difficulty"]])
    def source_files(scene_name, split, difficulty):
        if difficulty == "Mixed":
            return [
                f"{scene_name}/Easy/data_{split}.pkl",
                f"{scene_name}/Hard/data_{split}.pkl",
            ]
        return [f"{scene_name}/{difficulty}/data_{split}.pkl"]

    files = {
        scene: {
            "train": [path for difficulty in train_difficulties
                       for path in source_files(scene, "train", difficulty)],
            "val": [path for difficulty in val_difficulties
                     for path in source_files(scene, "val", difficulty)],
            "test": [path for difficulty in test_difficulties
                      for path in source_files(scene, "test", difficulty)],
        }
        for scene in args.scenes
    }
    manifest = {
        "protocol": args.protocol,
        "dataset": "TrajGap" if protocol_cfg.get("dataset") == "trajgap" else "TrajImpute",
        "data_source": {
            "type": "official_release",
            "root": str(data_root),
            "difficulty": protocol_cfg["difficulty"],
            "train_difficulties": train_difficulties,
            "val_difficulties": val_difficulties,
            "test_difficulties": test_difficulties,
            "train_sampling_scheme": protocol_cfg.get("train_sampling_scheme", "natural"),
            "train_sampling_seed": args.train_sampling_seed,
            "direct_prediction": True,
            "is_trajimpute_clean_split": False,
        },
        "data_root": str(data_root),
        "scenes": args.scenes,
        "files": files,
        "git_revision": get_git_revision(),
        "python": sys.executable,
        "env": env_info(),
        "model_version": args.variant,
        "encoder_type": VARIANTS[args.variant].get("encoder_type", "m0"),
        "encoder_source": "GAST Encoder" if VARIANTS[args.variant].get("encoder_type") in {"gast", "utenc"} else "M0",
        "encoder_config": {
            key: value for key, value in VARIANTS[args.variant].items()
            if key.startswith(("gast_", "utenc_"))
        },
        "model_config": _model_config(args),
        "scene_partition": bool(getattr(args, "allow_scene_partition", False)),
        "backbone": {"bimamba": args.bimamba},
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "precision": args.precision,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer_policy": OPTIMIZER_POLICY,
        "train_sampling_scheme": protocol_cfg.get("train_sampling_scheme", "natural"),
        "train_sampling_seed": args.train_sampling_seed,
        "sampling_definition": {
            "natural": "TrajGap Mixed train dataset with official Easy/Hard source samples concatenated and shuffled",
        },
        "monitor": checkpoint_monitor(args.K),
        "K": args.K,
        "initialization_checkpoint": args.init_checkpoint,
        "training_mode": "retrain_missing_data",
        "zero_missing_only": False,
        "gpu": args.gpu,
        "folds": args.folds,
        "started": datetime.now().isoformat(timespec="seconds"),
        "experiment_class": "smoke" if args.smoke else (
            "screening" if args.screening else "confirmatory"),
        "output_root": str(args.output_root),
        "command": " ".join(sys.argv),
        "runner_contract_version": RESUME_CONTRACT_VERSION,
    }
    manifest["optimizer_signature"] = expected_optimizer_signature(manifest)
    manifest["resume_contract"] = make_resume_contract(manifest)
    return manifest


def scene_manifest(manifest, scene):
    result = dict(manifest)
    result["scene"] = scene
    result["scenes"] = [scene]
    result["files"] = {scene: manifest["files"][scene]}
    result["resume_contract"] = make_resume_contract(result)
    return result



def sh(cmd, log_path, env=None):
    pretty = " ".join(str(c) for c in cmd)
    print("+", pretty, flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n$ {pretty}\n")
        p = subprocess.Popen([str(c) for c in cmd], stdout=f,
                             stderr=subprocess.STDOUT, cwd=REPO, env=env)
        rc = p.wait()
    return rc


def _run_one_scene_locked(args, scene, protocol_cfg, manifest, gpu_env):
    if protocol_cfg["dataset"] != "trajgap":
        raise ValueError("run_one_scene 只接受 TrajGap protocol")
    tag = f"{args.variant}_{scene}_{args.protocol}_seed{args.seed}"
    if args.bimamba is False:
        tag += "_uni"
    run_dir = Path(args.output_root) / tag
    checkpoints_dir = run_dir / "train" / "checkpoints"
    _manifest = run_dir / "manifest.json"
    expected_manifest = scene_manifest(manifest, scene)
    optimizer_signature = manifest["optimizer_signature"]

    if run_dir.exists():
        if not _manifest.is_file():
            raise RuntimeError(
                f"resume contract mismatch: existing run has no manifest: {run_dir}; "
                "use a new output root"
            )
        existing_manifest = json.loads(_manifest.read_text())
        validate_resume_manifest(existing_manifest, expected_manifest)
        resume_checkpoint = latest_valid_resume_checkpoint(
            checkpoints_dir, expected_manifest, optimizer_signature
        )
        if resume_checkpoint is None:
            raise RuntimeError(
                f"existing run has no checkpoint passing contract/optimizer preflight: {run_dir}"
            )
        resume_checkpoint = rebase_checkpoint_callback_paths(
            resume_checkpoint, checkpoints_dir
        )
        print(f"[resume-ok] contract-validated checkpoint: {resume_checkpoint}", flush=True)
    else:
        run_dir.mkdir(parents=True)
        (run_dir / "train").mkdir()
        _manifest.write_text(json.dumps(expected_manifest, indent=2, ensure_ascii=False))
        resume_checkpoint = None

    log = run_dir / "runner.log"

    # 训练
    train_overrides = [
        f"scene={scene}",
        f"difficulty={protocol_cfg['difficulty']}",
        f"train_difficulties={protocol_cfg.get('train_difficulties', [protocol_cfg['difficulty']])}",
        f"val_difficulties={protocol_cfg.get('val_difficulties', [protocol_cfg['difficulty']])}",
        f"train_sampling_scheme={protocol_cfg.get('train_sampling_scheme', 'natural')}",
        f"train_sampling_seed={args.train_sampling_seed}",
        f"zero_missing_only={str(protocol_cfg['zero_missing_only']).lower()}",
        f"data_root={args.data_root}",
        f"seed={args.seed}",
        f"batch_size={args.batch_size}",
        f"num_workers={args.num_workers}",
        f"epochs={args.epochs}",
        f"monitor={checkpoint_monitor(args.K)}",
        f"precision={args.precision}",
        f"run_manifest_path={_manifest}",
        f"model.target.model.num_modes={args.K}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"bimamba={str(args.bimamba).lower()}",
        f"clean_suffix={protocol_cfg['suffix']}",
    ]
    train_overrides.append(f"model_version={args.variant}")
    train_overrides.append(f"model.target.model.num_modes={args.K}")
    for switch, value in VARIANTS[args.variant].items():
        train_overrides.append(
            f"model.target.model.{switch}={str(value).lower() if isinstance(value, bool) else value}"
        )
    if args.init_checkpoint is not None:
        train_overrides.append(f"pretrained_weights={args.init_checkpoint}")
        train_overrides.append("checkpoint=null")
    if args.smoke:
        train_overrides += [f"limit_train_batches={args.limit_batches}",
                            f"limit_val_batches={args.limit_batches}"]
    train_cmd = [PY, "-u", "train.py",
                 f"--config-name={CONFIG_NAME}"] + train_overrides + [
        f"hydra.run.dir={run_dir / 'train'}"]

    # 断点续训：Lightning may version last.ckpt as last-vN.ckpt after resume.
    # Re-discover the furthest valid checkpoint after every failed attempt.
    rc = 1
    for attempt in range(1, getattr(args, "max_train_retries", 6) + 1):
        cmd = list(train_cmd)
        resume_checkpoint = latest_valid_resume_checkpoint(
            checkpoints_dir, expected_manifest, optimizer_signature
        )
        if attempt > 1 and resume_checkpoint is None:
            print("[resume-stop] no contract-valid checkpoint after failed attempt", flush=True)
            break
        if resume_checkpoint is not None:
            resume_checkpoint = rebase_checkpoint_callback_paths(
                resume_checkpoint, checkpoints_dir
            )
            cmd.append(f"checkpoint={resume_checkpoint}")
            print(f"[resume] attempt={attempt} from {resume_checkpoint}", flush=True)
        rc = sh(cmd, log, env=gpu_env)
        if rc == 0:
            break
        print(f"[train-failed] attempt={attempt} rc={rc}", flush=True)
        if latest_valid_resume_checkpoint(
                checkpoints_dir, expected_manifest, optimizer_signature) is None:
            break  # 无断点可续，放弃
    if rc != 0:
        return {"scene": scene, "status": "train_failed", "dir": str(run_dir), "rc": rc}

    # checkpoint：严格按最终输出头的验证指标选点，不使用最后 epoch。
    try:
        best_epoch, best_val, ckpt = select_best_checkpoint(
            run_dir / "train", checkpoint_monitor(args.K)
        )
    except RuntimeError as error:
        return {
            "scene": scene,
            "status": "checkpoint_selection_failed",
            "dir": str(run_dir),
            "error": str(error),
        }

    # 评估：同一 checkpoint 只评估统一的 Mixed 测试集。
    test_difficulties = protocol_cfg.get("test_difficulties", [protocol_cfg["difficulty"]])
    evaluations = []
    for test_difficulty in test_difficulties:
        eval_root = run_dir / "eval" / test_difficulty
        existing = prepare_eval_output(
            eval_root, scene, test_difficulty, args.variant, args.seed
        )
        if existing is not None:
            print(f"[eval-reuse] {scene}/{test_difficulty} 已有有效结果，跳过重跑", flush=True)
            evaluations.append({
                "difficulty": test_difficulty,
                "status": "complete",
                "rc": 0,
                "reused": True,
                "output_root": str(eval_root),
            })
            continue
        eval_cmd = [
            PY, "-u", str(EVAl_SCRIPT),
            "--data-root", args.data_root,
            "--scene", scene,
            "--difficulty", test_difficulty,
            "--split", "test",
            "--variant", args.variant,
            "--K", str(args.K),
            "--seed", str(args.seed),
            "--checkpoint", str(ckpt),
            "--output-root", str(run_dir / "eval" / test_difficulty),
            "--bimamba" if args.bimamba else "--no-bimamba",
        ]
        if protocol_cfg["zero_missing_only"]:
            eval_cmd += ["--zero-missing-only"]
        if args.smoke:
            eval_cmd += ["--max-batches", str(args.limit_batches)]
        rc_eval = sh(eval_cmd, log, env=gpu_env)
        evaluations.append({
            "difficulty": test_difficulty,
            "status": "complete" if rc_eval == 0 else "eval_failed",
            "rc": rc_eval,
            "output_root": str(run_dir / "eval" / test_difficulty),
        })
    status = "complete" if all(item["status"] == "complete" for item in evaluations) else "eval_failed"
    return {
        "scene": scene,
        "status": status,
        "dir": str(run_dir),
        "ckpt": str(ckpt),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "monitor": checkpoint_monitor(args.K),
        "K": args.K,
        "test_difficulties": test_difficulties,
        "evaluations": evaluations,
        "training_mode": "retrain_missing_data",
    }


def run_one_scene(args, scene, protocol_cfg, manifest, gpu_env):
    """Serialize one scene run under an exclusive output-directory lock."""
    tag = f"{args.variant}_{scene}_{args.protocol}_seed{args.seed}"
    if args.bimamba is False:
        tag += "_uni"
    run_dir = Path(args.output_root) / tag
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    with output_directory_lock(run_dir):
        return _run_one_scene_locked(args, scene, protocol_cfg, manifest, gpu_env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=list(PROTOCOLS))
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--folds", nargs="+", default=None, choices=FOLDS,
                    help="显式替代 --scenes 的等价写法（ETH -> ETH-M）")
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    epochs_default = 100
    ap.add_argument("--epochs", type=int, default=epochs_default)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-sampling-seed", type=int, default=None,
                    help="sampler seed；默认跟随 --seed")
    ap.add_argument("--K", type=int, default=DIRECT_NUM_MODES, choices=[DIRECT_NUM_MODES])
    ap.add_argument("--bimamba", action=argparse.BooleanOptionalAction, default=False,
                    help="主链固定单向(2026-09-12裁定)；旗标仅作用于encoder")
    ap.add_argument("--output-root", default="outputs/trajgap_bench")
    ap.add_argument(
        "--init-checkpoint",
        default=None,
        help="optional M0 initialization checkpoint loaded before training",
    )
    ap.add_argument("--allow-scene-partition", action="store_true",
                    help="allow disjoint subsets of the five scenes for coordinated GPU lanes")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-train-retries", type=int, default=6,
                    help="训练被杀后自动断点续训的最大重试次数")
    ap.add_argument("--screening", action="store_true",
                    help="标记为筛选实验（正式非确认性）")
    ap.add_argument("--limit-batches", type=int, default=2, help="smoke 用")
    args = ap.parse_args()
    if args.train_sampling_seed is None:
        args.train_sampling_seed = args.seed

    args.scenes = resolve_scenes(args.scenes, args.folds)
    args.folds = args.folds or []

    if args.K != DIRECT_NUM_MODES:
        raise SystemExit("TrajGap-Bench 正式重训协议固定 K=20")

    # model_version 标签（仅 output 命名）；模型开关由 VARIANTS 唯一映射。
    args.model_version_num = {"M0": "0", "GAST-PF": "G", "UT-ENC": "U"}[args.variant]

    protocol_cfg = PROTOCOLS[args.protocol]
    if args.protocol != "mixed-direct":
        raise SystemExit("正式实验只允许 TrajGap-Bench mixed-direct")
    if protocol_cfg["dataset"] != "trajgap":
        raise SystemExit("正式实验必须使用 TrajGap 数据适配器")
    if protocol_cfg["zero_missing_only"]:
        raise SystemExit("TrajGap-Bench 正式重训禁止 zero_missing_only")
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    if args.init_checkpoint is not None:
        args.init_checkpoint = str(Path(args.init_checkpoint).expanduser().resolve())
        if not Path(args.init_checkpoint).is_file():
            raise SystemExit(f"initialization checkpoint not found: {args.init_checkpoint}")
    if getattr(args, "paired_init", None) is not None:
        raise SystemExit("--paired-init was retired; use --init-checkpoint")

    gpu_env = dict(os.environ)
    gpu_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu_env["PYTHONNOUSERSITE"] = "1"
    gpu_env["PYTHONPATH"] = "."

    manifest = build_manifest(args, protocol_cfg)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    results = []
    for scene in args.scenes:
        try:
            r = run_one_scene(args, scene, protocol_cfg, manifest, gpu_env)
        except Exception:
            err = traceback.format_exc()
            print(err, file=sys.stderr)
            r = {"scene": scene, "status": "exception", "traceback": err}
        results.append(r)
        print(f"[{scene}] {r['status']}", flush=True)

    with open(Path(args.output_root) / f"runs_{args.variant}_{args.protocol}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""等待在跑的 UT-ENC（legacy 命名）补齐实验四个场景产出 results.json 后退出，
并输出与其他对比模型（MoFlow、SocialLSTM）的逐场景对照。

判定口径：只与 benchmark 中的其他对比模型比较，不与 M0 自身比较。

用途：把静默数小时的多车道训练链变成一次完成回调。
只读：轮询结果文件，不写入实验目录。
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/unitraj_encoder_bench_seed2024"
MOFLOW_ROOT = Path("/home/lbh/MoFlow/results_trajimpute_mixed")
SOCIAL_ROOT = Path("/home/lbh/SocialLSTM/trajgap_bench/artifacts/sociallstm_bench")

VARIANT = "UT-ENC"

# 已完成场景的固定路径（ETH-M 在单场景输出根下）
EXTRA = {
    "ETH-M": REPO / ("outputs/unitraj_encoder_single_scene_seed2024/"
                     "UT-ENC_ETH-M_mixed-direct_seed2024_uni"),
}

EXPECTED = {
    "HOTEL-M": ROOT / "lane_a/UT-ENC_HOTEL-M_mixed-direct_seed2024_uni",
    "ZARA2-M": ROOT / "lane_b/UT-ENC_ZARA2-M_mixed-direct_seed2024_uni",
    "ZARA1-M": ROOT / "lane_b/UT-ENC_ZARA1-M_mixed-direct_seed2024_uni",
    "UNIV-M": ROOT / "lane_d/UT-ENC_UNIV-M_mixed-direct_seed2024_uni",
}

SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]

MOFLOW_DIRS = {
    "ETH-M": "eth", "HOTEL-M": "hotel", "UNIV-M": "univ",
    "ZARA1-M": "zara1", "ZARA2-M": "zara2",
}

POLL_SECONDS = 300
MAX_HOURS = 40


def results_path(run_dir: Path, scene: str) -> Path:
    return (run_dir / "eval/Mixed" /
            f"{scene}_Mixed_test_{VARIANT}_seed2024_ckpt/results.json")


def overall(path: Path):
    """返回 (n, minADE20, minFDE20, MR)；缺文件返回 None。"""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    o = payload["results"]["overall"]
    return (o["n"], o["minADE_K"], o["minFDE_K"], o["MR"])


def ut_results():
    out = {}
    for scene, run_dir in {**EXTRA, **EXPECTED}.items():
        got = overall(results_path(run_dir, scene))
        if got:
            out[scene] = got
    return out


def moflow_results():
    out = {}
    for scene, sub in MOFLOW_DIRS.items():
        got = overall(MOFLOW_ROOT / sub / "mixed" / "results.json")
        if got:
            out[scene] = got
    return out


def social_results():
    out = {}
    for scene in SCENES:
        d = SOCIAL_ROOT / f"SocialLSTM_{scene}_mixed-direct_seed2024_uni"
        hits = sorted(d.glob("**/results.json"))
        if hits:
            got = overall(hits[0])
            if got:
                out[scene] = got
    return out


def report() -> str:
    ut = ut_results()
    mf = moflow_results()
    so = social_results()
    lines = ["=== UT-ENC vs 其他对比模型（Mixed-test, K=20, minFDE20）==="]
    lines.append(f"{'场景':9s} {'n':>8s} | {'UT-ENC':>9s} {'MoFlow':>9s} {'差':>8s} "
                 f"{'SocialLSTM':>11s} {'差':>8s}")
    win_mf = win_so = total = 0
    for scene in SCENES:
        if scene not in ut:
            lines.append(f"{scene:9s} {'—':>8s} | 尚未评估")
            continue
        total += 1
        n, _ua, uf, _umr = ut[scene]
        row = f"{scene:9s} {n:8d} | {uf:9.4f}"
        if scene in mf:
            m = mf[scene][2]
            d = 100 * (uf - m) / m
            row += f" {m:9.4f} {d:+7.1f}%"
            win_mf += d < 0
        else:
            row += f" {'—':>9s} {'—':>8s}"
        if scene in so:
            s = so[scene][2]
            d = 100 * (uf - s) / s
            row += f" {s:11.4f} {d:+7.1f}%"
            win_so += d < 0
        else:
            row += f" {'—':>11s} {'—':>8s}"
        lines.append(row)
    lines.append(f"汇总: 已评估 {total}/5 场景；"
                 f"优于 MoFlow {win_mf}/{total}，优于 SocialLSTM {win_so}/{total}")
    return "\n".join(lines)


def main() -> int:
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        pending = [
            scene for scene, run_dir in EXPECTED.items()
            if not results_path(run_dir, scene).is_file()
        ]
        if not pending:
            print("ALL_LANES_DONE")
            print(report())
            return 0
        print(f"[waiting] pending={pending}", flush=True)
        time.sleep(POLL_SECONDS)
    print("TIMEOUT")
    print(report())
    return 1


if __name__ == "__main__":
    sys.exit(main())

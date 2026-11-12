"""TrajGap-Bench 结果汇总。

扫描 TrajGap-Bench 输出根目录下的结果文件，
按 scene/seed/variant 聚合 minFDE20 等指标，输出：
  summary.json / summary.csv / comparison.md

只纳入 TrajGap-Bench 和 K=20；历史实验结果不会进入正式汇总。
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METRICS = ["minADE20", "minFDE20"]
SCENES = ["ETH-M", "HOTEL-M", "UNIV-M", "ZARA1-M", "ZARA2-M"]
FORMAL_K = 20
CANONICAL_PROTOCOL = "TrajGap-Bench"
CANONICAL_PROTOCOL_VALUES = {"trajgap-bench", CANONICAL_PROTOCOL}


def protocol_label(manifest, meta):
    """Return the formal benchmark protocol label."""
    raw = manifest.get("protocol") or meta.get("protocol")
    if raw in CANONICAL_PROTOCOL_VALUES:
        return CANONICAL_PROTOCOL
    return raw or "?"


def collect(root: Path):
    rows = []
    for res_file in sorted(root.rglob("results.json")):
        payload = json.loads(res_file.read_text())
        meta = payload["meta"]
        ov = payload["results"]["overall"]
        run_dir = res_file.parents[2]
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        row = {
            "variant": meta["variant"],
            "scene": meta["scene"],
            "protocol": protocol_label(manifest, meta),
            "seed": meta["seed"],
            "K": meta["K"],
            "n": ov.get("n"),
            "ckpt": meta.get("checkpoint", {}).get("path") if isinstance(meta.get("checkpoint"), dict) else None,
            "run_dir": str(run_dir),
        }
        for m in METRICS:
            legacy_key = {"minADE20": "minADE_K", "minFDE20": "minFDE_K"}[m]
            row[m] = ov.get(m, ov.get(legacy_key))
        rows.append(row)
    return rows


def agg(rows, key_fields):
    """按 key_fields 聚合：均值/标准差/最差/最好。"""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(r)
    out = []
    for key, grp in sorted(groups.items()):
        entry = dict(zip(key_fields, key))
        entry["n_runs"] = len(grp)
        entry["scenes"] = ",".join(g["scene"] for g in grp)
        for m in METRICS:
            vals = [g[m] for g in grp if g[m] is not None]
            if vals:
                entry[f"{m}_mean"] = sum(vals) / len(vals)
                entry[f"{m}_min"] = min(vals)
                entry[f"{m}_max"] = max(vals)
                if len(vals) > 1:
                    mu = entry[f"{m}_mean"]
                    entry[f"{m}_std"] = (sum((v - mu) ** 2 for v in vals)
                                         / (len(vals) - 1)) ** 0.5
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/trajgap_bench")
    args = ap.parse_args()
    root = Path(args.root)
    all_rows = collect(root)
    rows = [row for row in all_rows
            if row["K"] == FORMAL_K and row["protocol"] == CANONICAL_PROTOCOL]
    excluded = [
        {
            "run_dir": row["run_dir"],
            "K": row["K"],
            "protocol": row["protocol"],
            "reason": (
                f"TrajGap-Bench formal summary requires protocol={CANONICAL_PROTOCOL} "
                f"and K={FORMAL_K}"
            ),
        }
        for row in all_rows
        if row["K"] != FORMAL_K or row["protocol"] != CANONICAL_PROTOCOL
    ]
    if not rows:
        if excluded:
            print(f"no K={FORMAL_K} results found; excluded {len(excluded)} legacy rows")
        else:
            print("no results found")
        return

    # 明细
    detail_path = root / "summary.json"
    by_variant = agg(rows, ["protocol", "variant", "seed", "K"])
    payload = {
        "formal_K": FORMAL_K,
        "rows": rows,
        "excluded_legacy_rows": excluded,
        "by_variant": by_variant,
    }
    detail_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    # CSV
    csv_path = root / "summary.csv"
    fields = ["variant", "scene", "protocol", "seed", "K", "n"] + METRICS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # comparison.md：明细与聚合表
    md = ["# TrajGap-Bench 模型重训结果汇总\n"]
    md.append("数据来源：`" + str(root) + "`；训练与评估均为 direct 重训，K=20；")
    md.append("训练、验证和测试均使用 TrajGap 固定划分；所有误差指标越低越好。\n")

    md.append("## 各版本×场景×seed 明细\n")
    md.append("| variant | scene | protocol | seed | n | minADE20 | minFDE20 |")
    md.append("|---|---|---|---|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x["protocol"], x["variant"], x["seed"], SCENES.index(x["scene"]))):
        md.append(
            f"| {r['variant']} | {r['scene']} | {r['protocol']} | {r['seed']} | {r['n']} "
            f"| {r['minADE20']:.4f} | {r['minFDE20']:.4f} |")

    md.append("\n## 聚合（protocol × variant × seed × K）\n")
    md.append("| protocol | variant | seed | K | runs | minADE20均值 | minADE20标准差 | minFDE20均值 | minFDE20标准差 |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for e in by_variant:
        md.append(f"| {e['protocol']} | {e['variant']} | {e['seed']} | {e['K']} | {e['n_runs']} "
                  f"| {e.get('minADE20_mean', float('nan')):.4f} "
                  f"| {e.get('minADE20_std', 0):.4f} "
                  f"| {e.get('minFDE20_mean', float('nan')):.4f} "
                  f"| {e.get('minFDE20_std', 0):.4f} |")

    (root / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[summary] {detail_path}")
    print(f"[summary] {csv_path}")
    print(f"[summary] {root / 'comparison.md'}")


if __name__ == "__main__":
    main()

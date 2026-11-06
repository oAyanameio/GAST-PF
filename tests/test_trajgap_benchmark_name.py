import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.训练与评估.run_trajimpute_experiments import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
)
from scripts.结果分析.summarize_trajimpute import agg, protocol_label


def test_trajgap_bench_is_the_only_new_protocol_identifier():
    assert DEFAULT_PROTOCOL == "trajgap-bench"
    assert set(PROTOCOLS) == {"trajgap-bench"}
    assert PROTOCOLS["trajgap-bench"]["difficulty"] == "Mixed"


def test_summary_normalizes_canonical_protocol_metadata():
    assert protocol_label({"protocol": "trajgap-bench"}, {}) == "TrajGap-Bench"
    assert protocol_label({"protocol": "TrajGap-Bench"}, {}) == "TrajGap-Bench"


def test_summary_keeps_legacy_protocol_compatibility():
    assert protocol_label({"protocol": "mixed-direct"}, {}) == "TrajGap-Bench"


def test_summary_reads_protocol_from_results_when_manifest_is_missing():
    assert protocol_label({}, {"protocol": "mixed-direct"}) == "TrajGap-Bench"


def test_summary_aggregation_does_not_require_legacy_backbone_metadata():
    row = {
        "protocol": "TrajGap-Bench",
        "variant": "GAST",
        "seed": 2024,
        "K": 20,
        "scene": "ETH-M",
        "minADE_K": 0.2,
        "minFDE_K": 0.4,
        "ADE@1": 0.3,
        "FDE@1": 0.5,
    }

    result = agg([row], ["protocol", "variant", "seed", "K"])

    assert len(result) == 1
    assert result[0]["minFDE_K_mean"] == 0.4
    assert "backbone" not in result[0]

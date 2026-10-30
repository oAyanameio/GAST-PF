import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.训练与评估.run_trajimpute_experiments import (
    DEFAULT_PROTOCOL,
    PROTOCOLS,
)
from scripts.结果分析.summarize_trajimpute import protocol_label


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

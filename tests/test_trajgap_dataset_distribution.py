import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs.tools import gen_trajgap_dataset_distribution as distribution


def test_collect_counts_uses_trajgap_adapter_for_each_split(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, data_root, scene, split):
            calls.append((data_root, scene, split))
            self.sample_count = (distribution.SCENES.index(scene) + 1) * {
                'train': 1, 'test': 2, 'val': 3,
            }[split]

        def __len__(self):
            return self.sample_count

    monkeypatch.setattr(distribution, 'TrajGapDataset', FakeDataset)
    counts = distribution.collect_counts('/data/trajgap')

    assert counts['ETH-M'] == {'train': 1, 'test': 2, 'val': 3}
    assert len(calls) == len(distribution.SCENES) * len(distribution.SPLIT_ORDER)
    assert calls[0] == ('/data/trajgap', 'ETH-M', 'train')
    assert calls[-1] == ('/data/trajgap', 'ZARA2-M', 'val')
    assert distribution.split_totals(counts) == {
        'train': 15, 'test': 30, 'val': 45,
    }


def test_distribution_outputs_include_png_pdf_and_count_table(tmp_path):
    counts = {
        scene: {
            'train': index + 1,
            'test': 2 * (index + 1),
            'val': 3 * (index + 1),
        }
        for index, scene in enumerate(distribution.SCENES)
    }
    outdir = tmp_path / 'figures'
    report_path = tmp_path / 'results' / 'distribution.md'

    outputs = distribution.plot_distribution(counts, outdir)
    report = distribution.render_markdown(counts, outdir, report_path)

    assert {path.suffix for path in outputs} == {'.png', '.pdf'}
    assert all(path.is_file() for path in outputs)
    assert '| **Total** | **15** | **30** | **45** | **90** |' in report
    assert 'TrajGap 的 train、test 和 validation 划分' in report
    assert "TrajGap" in report
    assert 'Easy' not in report
    assert 'Hard' not in report

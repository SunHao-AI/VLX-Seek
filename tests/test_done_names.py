"""测试 _collect_done_names：全局去重扫描已有输出与 shard 文件。"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "distill"))

import generate_pseudo_labels as gpl


def _write_shard(path: Path, names: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": [{"id": i, "file_name": n} for i, n in enumerate(names)], "annotations": []}, f)


def test_collect_done_names_merges_and_tolerates_bad_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good1 = tmp / "out.shard0.json"
        good2 = tmp / "out.shard1.json"
        broken = tmp / "out.shard2.json"
        missing = tmp / "out.shard3.json"
        _write_shard(good1, ["a.jpg", "b.jpg"])
        _write_shard(good2, ["b.jpg", "c.jpg"])
        broken.write_text("{ 不是合法 json", encoding="utf-8")

        done = gpl._collect_done_names([str(good1), str(good2), str(broken), str(missing)])
        assert done == {"a.jpg", "b.jpg", "c.jpg"}, done


if __name__ == "__main__":
    test_collect_done_names_merges_and_tolerates_bad_files()
    print("test_done_names OK")

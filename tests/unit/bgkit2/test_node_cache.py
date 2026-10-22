import json

import pytest
import torch
from safetensors.torch import save_file

from bgkit2.data.node_cache import NodeCache


def test_single_file_layouts(tmp_path):
    f = str(tmp_path / "c.safetensors")
    save_file({"o/r/reps": torch.ones(3, 4), "o/r/mask": torch.ones(3, dtype=torch.bool),
               "o/s//pkg/reps": torch.zeros(2, 4), "o/s///reps": torch.full((5, 4), 2.0)}, f)
    c = NodeCache(f)
    assert c.repos == {"o/r", "o/s"} and len(c) == 3
    r, m = c.get("o/r")                      # root-only layout: node "" is the root
    assert r.shape == (3, 4) and m.all()
    assert c.get("o/s", "pkg")[0].shape == (2, 4) and c.get("o/s", "")[0].shape == (5, 4)
    assert ("o/s", "pkg") in c and sorted(c.nodes_of("o/s")) == ["", "pkg"]
    with pytest.raises(KeyError):
        c.get("o/s", "nope")


def test_sharded_directory_is_read_lazily(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    save_file({"a/b///reps": torch.ones(2, 4), "a/b//src/reps": torch.zeros(1, 4)},
              str(d / "part-0000.safetensors"))
    save_file({"c/d///reps": torch.full((3, 4), 7.0)}, str(d / "part-0001.safetensors"))
    with open(d / "index.jsonl", "w") as fh:
        fh.write(json.dumps({"repo": "a/b", "shard": "part-0000.safetensors", "nodes": ["", "src"]}) + "\n")
        fh.write(json.dumps({"repo": "c/d", "shard": "part-0001.safetensors", "nodes": [""]}) + "\n")
    c = NodeCache(str(d), max_open=1, max_tensors=1)
    assert len(c) == 3 and c.repos == {"a/b", "c/d"}
    assert float(c.get("c/d")[0].mean()) == 7.0
    assert c.get("a/b", "src")[0].shape == (1, 4)       # evicts and reopens cleanly
    assert float(c.get("c/d")[0].mean()) == 7.0


def _shard_dir(d, repo, meta):
    d.mkdir()
    save_file({f"{repo}///reps": torch.ones(2, 4)}, str(d / "part-0000.safetensors"))
    (d / "index.jsonl").write_text(
        json.dumps({"repo": repo, "shard": "part-0000.safetensors", "nodes": [""]}) + "\n")
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d)


def test_several_caches_as_one(tmp_path):
    """The trainer / student server take ONE cache path; S2 mixes the repo_tree_v6 trees with
    per-commit SWE trees ("a:b"). Same build settings required; a key in both is an error."""
    import os

    a = _shard_dir(tmp_path / "a", "o/r", {"max_depth": 6})
    b = _shard_dir(tmp_path / "b", "o/r@0123456789ab", {"max_depth": 6})
    c = NodeCache(os.pathsep.join([a, b]))
    assert c.repos == {"o/r", "o/r@0123456789ab"} and c.meta["max_depth"] == 6
    assert c.get("o/r@0123456789ab")[0].shape == (2, 4)
    with pytest.raises(ValueError, match="max_depth"):
        NodeCache(os.pathsep.join([a, _shard_dir(tmp_path / "d", "x/y", {"max_depth": 3})]))
    with pytest.raises(ValueError, match="earlier cache"):
        NodeCache(os.pathsep.join([a, _shard_dir(tmp_path / "e", "o/r", {"max_depth": 6})]))

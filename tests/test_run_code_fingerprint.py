"""
tests/test_run_code_fingerprint.py
==================================
The code fingerprint of the run (scripts/run_code_fingerprint.py): the
write / check logic on a temporary tree (identical; changed without a record
fails; changed with a record passes; a new unrecorded file fails; a missing
file fails; CRLF and LF copies hash alike), and the committed fingerprint
passes on the working tree with the derived path set.
"""

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_code_fingerprint import (FINGERPRINT, check_fingerprint, hash_tree_files,  # noqa: E402
                                          load_fingerprint, path_set_from_tree, record_change,
                                          sha256_normalised, write_fingerprint)

RUN = {"started": "2026-10-01T22:16:39", "git_head": "0123456789abcdef0123456789abcdef01234567", "mode": "full"}


def _tree(tmp_path):
    files = {"src/a.py": b"x = 1\n", "src/b.py": b"y = 2\r\n", "phases/p.py": b"z = 3\n"}
    for p, data in files.items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(data)
    return sorted(files)


STATUSES = ("identical", "changed-and-recorded", "MISMATCH", "missing", "not in fingerprint")


def _status(lines, path):
    """The printed status of one path (line format: '  <status>  <path>   (<note>)')."""
    for line in lines:
        s = line.strip()
        for st in STATUSES:
            if s.startswith(st) and s[len(st):].split()[:1] == [path]:
                return st
    return None


def test_crlf_and_lf_hash_alike():
    assert sha256_normalised(b"a\r\nb\r\n") == sha256_normalised(b"a\nb\n")
    assert sha256_normalised(b"a\nb\n") != sha256_normalised(b"a\nb")
    assert sha256_normalised(b"") == sha256_normalised(b"")


def test_write_and_check_on_a_temporary_tree(tmp_path):
    root, paths = str(tmp_path), _tree(tmp_path)
    fp = str(tmp_path / "fp.json")
    write_fingerprint(fp, RUN, hash_tree_files(root, paths))
    data = load_fingerprint(fp)
    assert list(data) == ["purpose", "run", "algorithm", "files"] and data["run"] == RUN
    assert list(data["files"]) == paths and all(set(v) == {"sha256_at_run"} for v in data["files"].values())

    lines = []
    assert check_fingerprint(root, paths, fp, out=lines.append) == 0
    assert all(_status(lines, p) == "identical" for p in paths)

    # changed without a record -> fail
    (tmp_path / "src" / "a.py").write_bytes(b"x = 2\n")
    lines = []
    assert check_fingerprint(root, paths, fp, out=lines.append) == 1
    assert _status(lines, "src/a.py") == "MISMATCH"

    # changed with a record -> pass, and the sentence is printed
    entry = record_change(fp, "src/a.py", "x became 2; test_x guarantees the old outputs", root=root)
    assert entry["sha256_now"] == sha256_normalised(b"x = 2\n")
    lines = []
    assert check_fingerprint(root, paths, fp, out=lines.append) == 0
    assert _status(lines, "src/a.py") == "changed-and-recorded"
    assert any("test_x guarantees" in line for line in lines)

    # a recorded file changed again -> MISMATCH
    (tmp_path / "src" / "a.py").write_bytes(b"x = 3\n")
    assert check_fingerprint(root, paths, fp, out=lambda s: None) == 1
    (tmp_path / "src" / "a.py").write_bytes(b"x = 2\n")

    # a CRLF copy of an LF file (and the reverse) hashes alike
    (tmp_path / "phases" / "p.py").write_bytes(b"z = 3\r\n")
    (tmp_path / "src" / "b.py").write_bytes(b"y = 2\n")
    assert check_fingerprint(root, paths, fp, out=lambda s: None) == 0

    # a new unrecorded file -> fail; recorded as added -> pass
    (tmp_path / "src" / "new.py").write_bytes(b"n = 1\n")
    more = paths + ["src/new.py"]
    lines = []
    assert check_fingerprint(root, more, fp, out=lines.append) == 1
    assert _status(lines, "src/new.py") == "not in fingerprint"
    record_change(fp, "src/new.py", "added after the run", root=root)
    assert load_fingerprint(fp)["files"]["src/new.py"]["sha256_at_run"] is None
    lines = []
    assert check_fingerprint(root, more, fp, out=lines.append) == 0
    assert _status(lines, "src/new.py") == "changed-and-recorded"

    # a missing file -> fail
    os.remove(tmp_path / "phases" / "p.py")
    lines = []
    assert check_fingerprint(root, more, fp, out=lines.append) == 1
    assert _status(lines, "phases/p.py") == "missing"


def test_committed_fingerprint_passes_on_the_working_tree():
    fp = os.path.join(_ROOT, FINGERPRINT)
    assert os.path.exists(fp)
    paths = path_set_from_tree(_ROOT)
    assert "src/evaluation.py" in paths and "phases/phase1.py" in paths and "src/__init__.py" in paths
    assert "scripts/run_phase1.py" in paths and "scripts/run_real_data.py" in paths
    assert "tests/test_accelerators.py" in paths                 # imported by scripts/run_phase0_tests.py
    assert "scripts/make_paper_tables.py" not in paths and "scripts/analyze_by_ltrue.py" not in paths
    assert "reproduce_all.py" not in paths
    lines = []
    assert check_fingerprint(_ROOT, paths, fp, out=lines.append) == 0, "\n".join(lines)
    data = load_fingerprint(fp)
    with open(os.path.join(_ROOT, "results", "run_manifest.json"), encoding="utf-8") as fh:
        M = json.load(fh)
    assert data["run"]["git_head"] == M["git_head"]["full"] and data["run"]["started"] == M["started"]
    assert data["run"]["mode"] == M["mode"] == "full"
    # the default CLI action is the check, and it passes
    r = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "run_code_fingerprint.py")],
                       cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FINGERPRINT CHECK: PASS" in r.stdout

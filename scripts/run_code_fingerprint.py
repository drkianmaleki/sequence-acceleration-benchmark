#!/usr/bin/env python3
"""
run_code_fingerprint.py
=======================
A fingerprint of the code that produced the committed results
(results/run_code_fingerprint.json), checkable by anyone with the tree.

The repository's rule is that results belong to the run recorded in
results/run_manifest.json.  The manifest names the commit of that run, but a
squash merge removes that commit from the public history; the fingerprint
survives it.  It records, for every file whose code enters a result, the
SHA-256 of the file as it was at the run.  A file deliberately changed or
added after the run carries its current hash and one sentence saying what
changed and which test guarantees the old outputs; any other difference is a
mismatch.

    python scripts/run_code_fingerprint.py                    # check the working tree (default)
    python scripts/run_code_fingerprint.py --check
    python scripts/run_code_fingerprint.py --write-from-commit <hash>
    python scripts/run_code_fingerprint.py --write-from-tree  # the driver does this at the start of a full run
    python scripts/run_code_fingerprint.py --record <path> "<sentence>"   # mark a file changed after the run

Path set (derived, not typed): every *.py under src/ and phases/; every step
script of reproduce_all.STEPS that produces results (every step that is not
in reproduce_all.GENERATOR_STEPS, the table generators); and every project
module those step scripts import from outside src/ and phases/ (today
tests/test_accelerators.py, imported by scripts/run_phase0_tests.py).

Hash: SHA-256 over the file bytes with CRLF normalised to LF, so a Windows
checkout and the git blob agree.  Paths are stored relative to the
repository root with forward slashes, sorted.

Check: one line per file -- identical / changed-and-recorded / MISMATCH /
missing / not in fingerprint -- and exit 0 only if nothing is mismatched,
missing or unrecorded.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FINGERPRINT = os.path.join("results", "run_code_fingerprint.json")
MANIFEST = os.path.join("results", "run_manifest.json")
CODE_DIRS = ("src", "phases")           # every *.py under these directories
ALGORITHM = ("sha256 over the file bytes with CRLF normalised to LF; paths relative to the repository "
             "root with forward slashes, sorted; the path set = every *.py under src/ and phases/, every "
             "result-producing step script of reproduce_all.STEPS and every project module those scripts "
             "import from outside src/ and phases/")
PURPOSE = ("Hashes of every code file that produced the results in this directory, as the files were at the run "
           "recorded in run_manifest.json; a file changed or added after the run carries sha256_now and "
           "changed_after_run, and `python scripts/run_code_fingerprint.py --check` must pass on the tree.")


# ── hashing ──────────────────────────────────────────────────────────────────
def sha256_normalised(data: bytes) -> str:
    """SHA-256 of the bytes with CRLF normalised to LF."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def hash_file(path: str) -> str:
    with open(path, "rb") as fh:
        return sha256_normalised(fh.read())


def _git(*args, binary=False):
    out = subprocess.check_output(["git"] + list(args), cwd=_ROOT, stderr=subprocess.DEVNULL)
    return out if binary else out.decode().strip()


def _git_exists(commit: str, path: str) -> bool:
    try:
        subprocess.check_output(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=_ROOT, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


# ── the path set ─────────────────────────────────────────────────────────────
def _step_scripts():
    """The result-producing step scripts of the driver (posix paths)."""
    from reproduce_all import STEPS, GENERATOR_STEPS
    return [script for _, script, _ in STEPS if script not in GENERATOR_STEPS]


def _imported_modules(source: str):
    """Dotted names of every module a Python source imports (import X / from X import ...)."""
    mods = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
    return mods


def _resolve_project_module(mod: str, exists):
    """
    The file of a dotted module name when it is a project module outside
    CODE_DIRS (`a/b.py` or `a/b/__init__.py` relative to the root), else None.
    """
    parts = mod.split(".")
    if parts[0] in CODE_DIRS:
        return None
    for cand in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
        if exists(cand):
            return cand
    return None


def _project_imports(start_paths, exists, read):
    """Transitive closure of the project modules outside CODE_DIRS imported by start_paths."""
    found, todo = [], list(start_paths)
    seen = set(todo)
    while todo:
        p = todo.pop(0)
        for mod in _imported_modules(read(p)):
            f = _resolve_project_module(mod, exists)
            if f is not None and f not in seen:
                seen.add(f)
                found.append(f)
                todo.append(f)
    return found


def path_set_from_tree(root: str = _ROOT):
    """The path set as it stands in the working tree (sorted posix paths)."""
    paths = []
    for d in CODE_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, d)):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                if fn.endswith(".py"):
                    paths.append(os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/"))
    steps = _step_scripts()
    missing = [s for s in steps if not os.path.exists(os.path.join(root, s))]
    if missing:
        raise FileNotFoundError(f"step scripts missing from the tree: {missing}")
    paths += steps

    def exists(p):
        return os.path.isfile(os.path.join(root, p))

    def read(p):
        with open(os.path.join(root, p), encoding="utf-8") as fh:
            return fh.read()

    paths += _project_imports(steps, exists, read)
    return sorted(set(paths))


def path_set_from_commit(commit: str):
    """The path set as it was at a commit (sorted posix paths)."""
    paths = []
    for line in _git("ls-tree", "-r", "--name-only", commit, "--", *CODE_DIRS).splitlines():
        if line.endswith(".py"):
            paths.append(line)
    steps = _step_scripts()
    missing = [s for s in steps if not _git_exists(commit, s)]
    if missing:
        raise FileNotFoundError(f"step scripts missing at {commit}: {missing}")
    paths += steps

    def exists(p):
        return _git_exists(commit, p)

    def read(p):
        return _git("show", f"{commit}:{p}", binary=True).decode("utf-8")

    paths += _project_imports(steps, exists, read)
    return sorted(set(paths))


# ── reading and writing ──────────────────────────────────────────────────────
def hash_tree_files(root: str, paths):
    return {p: hash_file(os.path.join(root, p)) for p in paths}


def hash_commit_files(commit: str, paths):
    return {p: sha256_normalised(_git("show", f"{commit}:{p}", binary=True)) for p in paths}


def load_manifest_run(manifest_path: str) -> dict:
    """The run block (started, git_head full hash, mode) copied from the manifest."""
    with open(manifest_path, encoding="utf-8") as fh:
        M = json.load(fh)
    return {"started": M.get("started"), "git_head": M.get("git_head", {}).get("full"), "mode": M.get("mode")}


def write_fingerprint(out_path: str, run: dict, hashes: dict, purpose: str = PURPOSE, algorithm: str = ALGORITHM) -> str:
    payload = {
        "purpose": purpose,
        "run": run,
        "algorithm": algorithm,
        "files": {p: {"sha256_at_run": h} for p, h in sorted(hashes.items())},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return out_path


def load_fingerprint(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_from_tree(root: str = _ROOT, manifest_path: str = None, out_path: str = None) -> str:
    """The driver's call at the start of a full run: hash the working tree's path set."""
    manifest_path = manifest_path or os.path.join(root, MANIFEST)
    out_path = out_path or os.path.join(root, FINGERPRINT)
    return write_fingerprint(out_path, load_manifest_run(manifest_path), hash_tree_files(root, path_set_from_tree(root)))


def record_change(fp_path: str, path: str, sentence: str, root: str = _ROOT) -> dict:
    """Mark a file as deliberately changed (or added) after the run: current hash + one sentence."""
    fp = load_fingerprint(fp_path)
    path = path.replace(os.sep, "/")
    entry = fp["files"].setdefault(path, {"sha256_at_run": None})
    entry["sha256_now"] = hash_file(os.path.join(root, path))
    entry["changed_after_run"] = sentence
    fp["files"] = dict(sorted(fp["files"].items()))
    with open(fp_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(fp, fh, indent=2)
        fh.write("\n")
    return entry


# ── the check ────────────────────────────────────────────────────────────────
def check_fingerprint(root: str, paths, fp_path: str, out=print) -> int:
    """
    Compare the working tree (the given path set) with the stored fingerprint.
    One line per file; returns 0 only if nothing is mismatched, missing or
    unrecorded.
    """
    fp = load_fingerprint(fp_path)
    files = fp.get("files", {})
    counts = {"identical": 0, "changed-and-recorded": 0, "MISMATCH": 0, "missing": 0, "not in fingerprint": 0}
    for p in sorted(set(files) | set(paths)):
        entry = files.get(p)
        full = os.path.join(root, p)
        if entry is None:
            status, note = "not in fingerprint", ""
        elif not os.path.isfile(full):
            status, note = "missing", ""
        else:
            h = hash_file(full)
            if entry.get("sha256_at_run") is not None and h == entry["sha256_at_run"]:
                status, note = "identical", ""
            elif entry.get("sha256_now") == h and entry.get("changed_after_run"):
                status = "changed-and-recorded"
                note = ("added after the run: " if entry.get("sha256_at_run") is None else "") + entry["changed_after_run"]
            else:
                status, note = "MISMATCH", "hash differs from sha256_at_run" + (" and from sha256_now" if entry.get("sha256_now") else "")
        counts[status] += 1
        out(f"  {status:<22} {p}" + (f"   ({note})" if note else ""))
    bad = counts["MISMATCH"] + counts["missing"] + counts["not in fingerprint"]
    out(f"  {len(set(files) | set(paths))} files: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    out("  FINGERPRINT CHECK: " + ("PASS" if bad == 0 else f"FAIL ({bad} file(s) mismatched, missing or unrecorded)"))
    return 0 if bad == 0 else 1


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Code fingerprint of the run that produced results/ (write or check).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="compare the working tree with the stored fingerprint (default)")
    g.add_argument("--write-from-commit", metavar="HASH", help="hash the path set as it is at that commit and write the fingerprint")
    g.add_argument("--write-from-tree", action="store_true", help="hash the working tree's path set and write the fingerprint")
    g.add_argument("--record", nargs=2, metavar=("PATH", "SENTENCE"), action="append",
                   help="mark PATH as deliberately changed or added after the run (stores sha256_now and the sentence)")
    p.add_argument("--fingerprint", default=os.path.join(_ROOT, FINGERPRINT))
    p.add_argument("--manifest", default=os.path.join(_ROOT, MANIFEST))
    args = p.parse_args(argv)

    if args.write_from_commit:
        commit = _git("rev-parse", args.write_from_commit)
        run = load_manifest_run(args.manifest)
        if run["git_head"] != commit:
            print(f"ERROR: {args.manifest} records git_head {run['git_head']}, not {commit}; a fingerprint must describe the run's own code")
            return 2
        paths = path_set_from_commit(commit)
        out = write_fingerprint(args.fingerprint, run, hash_commit_files(commit, paths))
        print(f"  wrote {os.path.relpath(out, _ROOT)}: {len(paths)} files hashed at {commit[:7]}")
        return 0
    if args.write_from_tree:
        out = write_from_tree(_ROOT, args.manifest, args.fingerprint)
        print(f"  wrote {os.path.relpath(out, _ROOT)}: {len(load_fingerprint(out)['files'])} files hashed from the working tree")
        return 0
    if args.record:
        for path, sentence in args.record:
            e = record_change(args.fingerprint, path, sentence)
            print(f"  recorded {path}: sha256_now {e['sha256_now'][:12]}...; {sentence}")
        return 0
    print(f"run_code_fingerprint.py --check: {os.path.relpath(args.fingerprint, _ROOT)} against the working tree")
    return check_fingerprint(_ROOT, path_set_from_tree(_ROOT), args.fingerprint)


if __name__ == "__main__":
    sys.exit(main())

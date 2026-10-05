"""The tables of the paper's main text are built from the generated fragments by released code."""
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "build_paper_tables.py")


def test_builder_reproduces_every_fragment_and_builds_the_main_text_tables(tmp_path):
    run = subprocess.run([sys.executable, SCRIPT, "--selftest", "--out", str(tmp_path)],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "TABLE BUILD: PASS" in run.stdout
    assert " 0 not" in run.stdout                       # every fragment passes through the engine unchanged
    spec = importlib.util.spec_from_file_location("build_paper_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = sorted(name for name in os.listdir(tmp_path) if name.startswith("tab_"))
    assert built == sorted(table["name"] + ".tex" for table in module.TABLES)

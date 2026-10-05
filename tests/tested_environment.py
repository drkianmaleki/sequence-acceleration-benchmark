"""
tests/tested_environment.py
===========================
Whether the running environment is the tested one of requirements-lock.txt:
the same operating system (platform.system()), the same Python major and
minor version, and the same versions of numpy and scipy.

Two tests assert exact equality of fitted numbers against pinned values
(tests/test_perturbations.py::test_shift_iqr_is_unchanged_pinned_values,
tests/test_real_roster.py::test_legacy_path_reproduces_the_committed_results_on_a_subset).
Least-squares fits with several parameters can differ in their last digits
between platforms and library versions, so those two tests run only when
is_tested_environment() is true and are skipped with SKIP_REASON otherwise.
Every other test is platform-independent.
"""

import os
import platform
import re
import sys

LOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements-lock.txt")
SKIP_REASON = "exact reproduction is asserted only in the tested environment (requirements-lock.txt)"
PINNED_PACKAGES = ("numpy", "scipy")


def tested_environment(lock_path=LOCK):
    """The tested environment as the lock file records it: {'system', 'python', 'numpy', 'scipy'}."""
    env = {}
    with open(lock_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"#\s*platform\.system\(\):\s*(\S+)", line)
            if m:
                env["system"] = m.group(1)
                continue
            m = re.match(r"#\s*python:\s*(\d+)\.(\d+)", line)
            if m:
                env["python"] = (int(m.group(1)), int(m.group(2)))
                continue
            m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line)
            if m and m.group(1) in PINNED_PACKAGES:
                env[m.group(1)] = m.group(2)
    return env


def running_environment():
    """The same fields for the interpreter running the tests."""
    env = {"system": platform.system(), "python": (sys.version_info[0], sys.version_info[1])}
    for name in PINNED_PACKAGES:
        try:
            env[name] = __import__(name).__version__
        except Exception:
            env[name] = None
    return env


def is_tested_environment(lock_path=LOCK):
    """True only when the operating system, the Python major.minor and the numpy and scipy versions
    all equal the tested ones; False when the lock file is missing or incomplete."""
    try:
        tested = tested_environment(lock_path)
    except OSError:
        return False
    if any(k not in tested for k in ("system", "python", *PINNED_PACKAGES)):
        return False
    running = running_environment()
    return all(running[k] == tested[k] for k in ("system", "python", *PINNED_PACKAGES))

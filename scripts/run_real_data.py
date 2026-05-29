import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.datasets import run_real_data_experiment

if __name__ == '__main__':
    out_dir = os.path.join('results', 'real_data')
    run_real_data_experiment(out_dir=out_dir)
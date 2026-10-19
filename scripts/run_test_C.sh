#!/bin/bash
# Stream C of the final evaluation (MFRNP), started separately at low priority so that it uses idle CPU.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_budget.py test results/test_C.csv --methods MFRNP --budgets 10,20,40 --repeats 10 --lf 128 > results/test_C.log 2>&1
python3 scripts/run_budget.py test results/test_C.csv --methods MFRNP --budgets 80,160 --repeats 5 --lf 128 >> results/test_C.log 2>&1
echo "stream C done $(date)" >> results/test_runs.log

#!/bin/bash
# Post-hoc comparison baseline VRN (PROTOCOL 15:45): tune on validation, then the test runs with the BMLP draws.
# Two streams write separate files, which are appended to results/test_D.csv at the end.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/tune_nn.py VRN > results/tune_vrn.log 2>&1
rm -f results/test_vrn_a.csv results/test_vrn_b.csv
(python3 scripts/run_budget.py test results/test_vrn_a.csv --methods VRN --budgets 10,20 --repeats 20 --lf 128 > results/test_vrn_a.log 2>&1) &
(python3 scripts/run_budget.py test results/test_vrn_b.csv --methods VRN --budgets 40 --repeats 20 --lf 128 > results/test_vrn_b.log 2>&1 &&
 python3 scripts/run_budget.py test results/test_vrn_b.csv --methods VRN --budgets 80,160 --repeats 10 --lf 128 >> results/test_vrn_b.log 2>&1) &
wait
python3 -I -c "
import pandas as pd
d=pd.read_csv('results/test_D.csv'); a=pd.read_csv('results/test_vrn_a.csv'); b=pd.read_csv('results/test_vrn_b.csv')
assert list(a.columns)==list(d.columns)==list(b.columns)
pd.concat([d,a,b]).to_csv('results/test_D.csv', index=False); print(len(a), len(b))
"
echo "VRN done $(date)" >> results/test_runs.log

#!/bin/bash
# Final locked evaluation on the test set (run once). Launched automatically when data/test.jsonl is complete.
# Stream A and B run in parallel (one core each); MFRNP (stream C) runs afterwards.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while [ "$(wc -l < data/test.jsonl)" -lt 300 ]; do sleep 30; done
echo "test data complete $(date)" >> results/test_runs.log
A="MT,LF,RICH,BN-COR,COR,BN-AR1,AR1"
B="GP,BN-GP,BN-NARGP,NARGP,XGB,MLP,BMLP,TLNN,MFNN"
(
  python3 scripts/run_budget.py test results/test_A.csv --methods $A --budgets 10,20,40 --repeats 20 --lf 128 > results/test_A.log 2>&1
  python3 scripts/run_budget.py test results/test_A.csv --methods $A --budgets 80,160 --repeats 10 --lf 128 >> results/test_A.log 2>&1
  echo "stream A done $(date)" >> results/test_runs.log
) &
(
  python3 scripts/run_budget.py test results/test_B.csv --methods $B --budgets 10,20,40 --repeats 20 --lf 128 > results/test_B.log 2>&1
  while ! python3 -c "import json,sys; h=json.load(open('results/nn_hparams.json')); sys.exit(0 if '160' in h.get('MFNN',{}) else 1)"; do sleep 60; done
  python3 scripts/run_budget.py test results/test_B.csv --methods $B --budgets 80,160 --repeats 10 --lf 128 >> results/test_B.log 2>&1
  echo "stream B done $(date)" >> results/test_runs.log
) &
wait
python3 scripts/run_budget.py test results/test_C.csv --methods MFRNP --budgets 10,20,40 --repeats 10 --lf 128 > results/test_C.log 2>&1
python3 scripts/run_budget.py test results/test_C.csv --methods MFRNP --budgets 80,160 --repeats 5 --lf 128 >> results/test_C.log 2>&1
echo "stream C done $(date)" >> results/test_runs.log

#!/bin/bash
# Full reproduction of the study, stage by stage. Usage: bash reproduce.sh [stage ...]
# Stages: deps data verify select test posthoc numbers figs all   (default: all).
# Expected cost on one CPU core: data ~2.2 h, test runs ~5 h (MFRNP ~3 h more), everything else < 1 h.
set -e
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} EXT_DIR=${EXT_DIR:-external}
stages=${*:-all}
has() { [[ " $stages " == *" $1 "* || " $stages " == *" all "* ]]; }

if has deps; then
  pip install -r requirements.txt
  bash scripts/get_external.sh                       # MFRNP code, Mechanical MNIST, natural-fibre data set (pinned commits)
  python3 -m pytest tests -q
fi
if has data; then
  for s in pool val test lfx scatter ref; do python3 scripts/generate_data.py $s data/$s.jsonl 2; done
fi
if has verify; then
  python3 scripts/verify_solver.py results/verify_solver.jsonl
  python3 scripts/check_tolerance.py
  python3 scripts/bounds_stats.py
  python3 scripts/order_parameter.py results/order_parameter.json
  python3 scripts/figs/fig03_verify.py && python3 scripts/order_check.py
  python3 scripts/benchmark_fe.py results/fe_benchmark.json
fi
if has select; then                                  # model selection on the validation set (results/PROTOCOL.md)
  python3 scripts/run_budget.py val results/val_sel_128.csv --methods LF,COR,BN-COR --repeats 5 --ratio 1 --lf 128
  python3 scripts/run_budget.py val results/val_sel_64.csv  --methods LF,RICH,GP,BN-GP,COR,BN-COR --repeats 5 --ratio 1 --lf 64
  python3 scripts/run_budget.py val results/val_mf_128.csv  --methods BN-AR1,BN-NARGP --repeats 5 --lf 128
  python3 scripts/run_budget.py val results/val_mf_64.csv   --methods BN-AR1,BN-NARGP --repeats 5 --lf 64
  python3 scripts/tune_nn.py MLP,BMLP,XGB,MFNN,TLNN results/nn_hparams.json
fi
if has test; then                                    # locked test set, run once
  bash scripts/run_test.sh
  bash scripts/run_test_C.sh
fi
if has posthoc; then                                 # analyses added after the test (all marked in results/PROTOCOL.md)
  python3 scripts/run_budget.py test results/test_D.csv --methods BN-LINCOR,BN-COR-X,MT-GP --budgets 10,20,40 --repeats 20 --lf 128
  python3 scripts/run_budget.py test results/test_D.csv --methods BN-LINCOR,BN-COR-X,MT-GP --budgets 80,160 --repeats 10 --lf 128
  bash scripts/run_vrn.sh
  bash scripts/run_derived_v2.sh
  python3 scripts/conformal.py
  python3 scripts/scatter_eval.py
fi
if has numbers; then
  python3 scripts/make_tables.py test
  for n in 10 20 40; do python3 scripts/bootstrap_test.py $n BN-COR results/boot_${n}_BN-COR.npz; done
  python3 scripts/bootstrap_test.py 10 BMLP results/boot_10_BMLP.npz; python3 scripts/bootstrap_test.py 20 BMLP results/boot_20_BMLP.npz
  python3 scripts/bootstrap_test.py 40 MFNN results/boot_40_MFNN.npz; python3 scripts/bootstrap_test.py summary
  python3 scripts/check_sec1_3.py
  python3 scripts/check_sec4.py
  bash scripts/run_predictions.sh
  python3 scripts/benchmark_cost.py results/cost_benchmark.json
  python3 scripts/benchmark_cost.py results/cost_benchmark.json --analytic-only && python3 scripts/make_table4.py
  python3 scripts/application.py
  python3 scripts/application_curve.py
  python3 scripts/run_external_nf.py final results/ext_nf_final.csv
  python3 scripts/run_external_mnist.py results/ext_mnist.csv
  python3 scripts/mnist_published.py results/mnist_published.json
  python3 scripts/check_sec48.py results/numbers_sec48.json
fi
if has figs; then
  for f in scripts/figs/fig*.py; do python3 $f; done   # Figs. 1-14 -> paper/figs/
fi
echo "done: $stages"

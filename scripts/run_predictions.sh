#!/bin/bash
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M=BN-COR,BN-AR1,BN-LINCOR,AR1,NARGP,GP,BN-GP,MFNN,TLNN,MLP,BMLP,XGB
python3 scripts/predictions.py 20 0 $M results/pred_test_nH20_rep0.npz test > results/pred.log 2>&1
python3 scripts/predictions.py 40 0 $M results/pred_test_nH40_rep0.npz test >> results/pred.log 2>&1
python3 scripts/predictions.py 160 0 BN-COR,BN-AR1 results/pred_scatter_nH160_rep0.npz scatter >> results/pred.log 2>&1
echo "predictions done $(date)" >> results/pred.log

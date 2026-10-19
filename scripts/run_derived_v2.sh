#!/bin/bash
# Recompute derived constants after the nu12 sign fix (PROTOCOL 16:04), two streams, then merge.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rm -f results/test_derived_v2_a.csv results/test_derived_v2_b.csv
python3 -I scripts/derived_v2.py MT,LF,RICH,BN-COR results/test_derived_v2_a.csv > results/derived_v2_a.log 2>&1 &
python3 -I scripts/derived_v2.py BMLP results/test_derived_v2_b.csv > results/derived_v2_b.log 2>&1 &
wait
python3 -I -c "
import pandas as pd
d=pd.concat([pd.read_csv('results/test_derived_v2_a.csv'), pd.read_csv('results/test_derived_v2_b.csv')])
d.to_csv('results/test_derived_v2.csv', index=False); print(len(d), 'rows; all reproduced:', bool(d.reproduced.all()))
"

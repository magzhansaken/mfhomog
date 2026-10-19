# mfhomog: solver-in-the-loop, bound-normalised prediction of the elastic constants of unidirectional composites

Code, data and scripts for the manuscript
*基于求解器在环校正的单向复合材料弹性常数小样本预测方法* (Small-sample prediction method for elastic constants of unidirectional composites based on solver-in-the-loop correction)
by T. Mukasheva, S. Zhuzbayev, A. Mukasheva and M. Sarsenbay (submitted to 机械工程学报 / Journal of Mechanical Engineering). The article DOI is added on acceptance.

本仓库包含论文全部程序、数据生成脚本、预先登记的试验协议与绘图脚本，所有数值结果均可由下述命令复现。

**Method in one sentence.** For a new microstructure one coarse-grid FFT homogenisation (N = 128, about 1/24 of the cost of the
high-fidelity solution) is run online, and a Gaussian process trained on 10–160 high-fidelity samples corrects it in a coordinate
normalised by the Hashin–Shtrikman bounds, so that every prediction and its interval lie inside the bounds. A parametric mode
(recursive co-kriging in the same coordinate) is provided for design-space scans without online solves.

## Quick start

```bash
pip install -r requirements.txt
python3 examples/quickstart.py      # trains the hybrid model on 20 samples and predicts a new random microstructure (~1 min)
```
Output: coarse-grid moduli, corrected prediction with its 95 % interval, the high-fidelity reference and the error (typically a few
tenths of a percent with 20 training samples), plus the derived engineering constants E2, nu23, E1, nu12.


## What is in the repository

| path | content |
|---|---|
| `mfhomog/fft_solver.py` | FFT-Galerkin homogenisation (anti-plane shear → G12; plane strain → k23, G23), CG solver |
| `mfhomog/fe_check.py` | independent periodic Q1 finite-element solver on the same pixel images (verification) |
| `mfhomog/rve.py` | random hard-disk fibre arrangements (36 fibres, Metropolis), rasterisation at any resolution |
| `mfhomog/micromech.py` | Hashin–Shtrikman bounds, Mori–Tanaka, Halpin–Tsai, Hill's exact connections |
| `mfhomog/surrogates.py` | GP, recursive AR1 co-kriging, NARGP (emukit), **BN-COR** (proposed hybrid), bound-normalised transform |
| `mfhomog/nn_models.py` | MLP, bound-preserving MLP, Voigt-Reuss Net (re-implemented), MFNN, transfer-learning NN, XGBoost baselines |
| `mfhomog/mfrnp_adapter.py` | adapter for the MFRNP reference code (Niu et al., ICML 2024; MIT) |
| `scripts/generate_data.py` | generation of all data sets (pool, val, test, lfx, scatter, ref) |
| `scripts/run_budget.py` | budget experiment (all methods × budgets × repeats) |
| `scripts/make_tables.py`, `scripts/bootstrap_test.py` | summary tables, paired Wilcoxon tests with Holm correction, bootstrap over test samples |
| `examples/quickstart.py` | minimal end-to-end example (train on 20 samples, predict a new microstructure) |
| `reproduce.sh` | staged reproduction of the whole study |
| `scripts/run_external_nf.py`, `scripts/run_external_mnist.py`, `scripts/mnist_published.py`, `scripts/check_sec48.py` | external data sets |
| `scripts/figs/` | one script per figure of the paper |
| `results/PROTOCOL.md` | pre-registered protocol (written before any validation/test result) and dated amendments |
| `data/*.jsonl` | generated data (one JSON line per RVE: inputs, seeds, moduli per resolution, CPU times) |

## Reproduce

`bash reproduce.sh` runs everything below stage by stage (`bash reproduce.sh data`, `... test`, `... figs`, ...). The individual commands:

```bash
pip install -r requirements.txt
EXT_DIR=external bash scripts/get_external.sh          # external code/data at the commits used (MFRNP, Mechanical MNIST, natural-fibre set)
export EXT_DIR=external
python3 -m pytest tests -q                                   # unit tests
for s in pool val test lfx scatter ref; do python3 scripts/generate_data.py $s data/$s.jsonl 2; done
python3 scripts/verify_solver.py results/verify_solver.jsonl   # FFT vs independent Q1 FE (Fig. 3b)
python3 scripts/check_tolerance.py                             # CG tolerance check (section 1.2)
python3 scripts/bounds_stats.py                                # HS bounds on the data, MT/HT identity (section 1.3)
python3 scripts/order_parameter.py results/order_parameter.json   # hexatic order of the arrangements (section 1.1)
python3 scripts/figs/fig03_verify.py && python3 scripts/order_check.py   # convergence order, HF error (section 1.4)
OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark_fe.py results/fe_benchmark.json   # FE vs FFT CPU time (section 1.4)
# selection on the validation set (see results/PROTOCOL.md), e.g.
python3 scripts/run_budget.py val results/val_sel_128.csv --methods LF,COR,BN-COR --repeats 5 --ratio 1 --lf 128
python3 scripts/tune_nn.py MLP,BMLP,XGB,MFNN,TLNN results/nn_hparams.json
# final evaluation on the locked test set (run once)
bash scripts/run_test.sh          # streams A and B
bash scripts/run_test_C.sh        # MFRNP
python3 scripts/run_budget.py test results/test_D.csv --methods BN-LINCOR --budgets 10,20,40 --repeats 20 --lf 128   # post-hoc ablation
python3 scripts/run_budget.py test results/test_D.csv --methods BN-LINCOR --budgets 80,160 --repeats 10 --lf 128
bash scripts/run_vrn.sh                                       # post-hoc baseline VRN (PROTOCOL, about 15:37)
bash scripts/run_derived_v2.sh                                # derived constants after the nu12 fix (PROTOCOL 16:04)
python3 scripts/make_tables.py test                            # Table 3, paired tests, paper/numbers_test.json
for n in 10 20 40; do python3 scripts/bootstrap_test.py $n BN-COR results/boot_${n}_BN-COR.npz; done; python3 scripts/bootstrap_test.py 10 BMLP results/boot_10_BMLP.npz; python3 scripts/bootstrap_test.py 20 BMLP results/boot_20_BMLP.npz; python3 scripts/bootstrap_test.py 40 MFNN results/boot_40_MFNN.npz; python3 scripts/bootstrap_test.py summary   # section 4.2
python3 scripts/check_sec1_3.py                           # numbers of sections 1-3 not stored elsewhere
python3 scripts/check_sec4.py                                 # numbers of section 4.1
python3 scripts/conformal.py                                  # split-conformal calibration of the hybrid intervals (section 4.5)
python3 scripts/scatter_eval.py                               # scatter set, all repeats and budgets (section 4.6, Fig. 11a)
bash scripts/run_predictions.sh                                # per-sample predictions for Figs. 8, 9, 11
python3 scripts/benchmark_cost.py results/cost_benchmark.json  # CPU times (Fig. 12)
python3 scripts/benchmark_cost.py results/cost_benchmark.json --analytic-only && python3 scripts/make_table4.py   # Table 4
python3 scripts/application.py                                 # application example (section 4.9)
python3 scripts/application_curve.py                        # E2-V_f curve, MT comparison and costs of section 4.9 (Fig. 14)
python3 scripts/run_external_nf.py final results/ext_nf_final.csv
python3 scripts/run_external_mnist.py results/ext_mnist.csv
python3 scripts/mnist_published.py results/mnist_published.json   # published CNN / TL-CNN errors from the authors' prediction files
python3 scripts/check_sec48.py results/numbers_sec48.json         # numbers of section 4.8
for f in scripts/figs/fig*.py; do python3 $f; done             # Figs. 1-14 (file number = figure number)
# the manuscript itself is not part of this repository; the figure scripts write to paper/figs/
```
Cost on the original machine (no GPU): the five modelling data sets cost about 2.2 CPU-hours at the single-core benchmark
rates of section 1.2 (`results/numbers_sec1_3.json`); the recorded wall-clock solve times of the original run (2 worker
processes on a loaded machine) sum to 11 h. The test runs took about 1.8 h of wall-clock time for streams A and B (in
parallel) and 2.8 h for MFRNP (stream C).
Results reproduce to about 1e-7 in relative terms; in rare cases the GP hyper-parameter optimisation ends in a slightly
different optimum (observed once: scatter set, BN-AR1, n_H = 160, repeat 1, k23 MAPE 0.289 % vs 0.285 % logged).
All random seeds are fixed in the scripts; design seeds and microstructure seeds differ between data sets.

## External data (not redistributed here)

* Short natural-fibre RVE dataset, CC BY 4.0: https://github.com/roysondsouza316/natural-fibre-rve-dataset-surrogate
* Mechanical MNIST transfer-learning data (Lejeune 2020; Lejeune & Zhao 2021), CC BY-SA 4.0 data / MIT code:
  https://github.com/elejeune11/Mechanical-MNIST-Transfer-Learning
* MFRNP reference code (MIT): https://github.com/Rose-STL-Lab/MFRNP

## Citation

If you use this code or data, please cite the article (see `CITATION.cff`); an archived release of this repository with a DOI is listed there.

## Licence

Code: MIT. Generated data in `data/` and results in `results/`: CC BY 4.0.

# Pre-registered protocol (written before any validation or test result was seen)

Date: 16.10.2026, 21:05 (local).

## Data roles
- pool (400 points): training pool; every repeat draws its high-fidelity (HF) training set from it.
- lfx (first 800 of 1600 planned Sobol points, LF only; capped at 21:40 for compute time, before any multi-fidelity result was seen): additional low-fidelity data for multi-fidelity models (nested design: LF also at the HF points).
- val (100 points): model selection and hyper-parameter tuning only.
- test (300 points): locked; evaluated once, after all choices below are frozen.
- scatter (10 points x 8 microstructures): irreducible realisation scatter of the HF target.
- ref (12 points at 256/512/1024): accuracy of the HF definition (Richardson extrapolation).
Microstructure seeds and design seeds differ between all sets.

## Selection of the proposed variant (on val, repeats 0-4, budgets 10/20/40/80/160, LF:HF = 4)
- Parametric candidates (no solve at query time): BN-GP, BN-AR1@64, BN-AR1@128, BN-NARGP@64, BN-NARGP@128.
- Hybrid candidates (one low-fidelity solve at query time): BN-COR@64, BN-COR@128.
- Criterion: lowest mean over budgets of mape_mean3 (mean MAPE of G12, G23, k23). Differences within 5 % -> the cheaper
  variant (fewer data and query seconds) is chosen.
- The hybrid mode is reported as a second operating mode only if its selected variant beats on val both the selected
  parametric variant and the purely numerical baseline RICH (Richardson 64/128 at the query).
- If a component (multi-fidelity, hybrid correction) does not improve on val, it is removed from the method, the title
  and the claims.

## Baselines (all run in the same protocol on test)
MT, HT (analytical), LF and RICH (numerical only), GP (Kriging on log targets), AR1 (recursive co-kriging), NARGP,
MLP (BP), BMLP (bound-preserving MLP, output layer as in the Voigt-Reuss Net), XGB, MFNN (Meng & Karniadakis 2020),
TLNN (transfer learning), MFRNP (Niu et al., ICML 2024). NN/XGB hyper-parameters chosen on val (repeat 0).

## Final evaluation (test, once)
20 repeats per budget; metrics MAPE/RMSE/R2/max APE per target, derived E2 and nu23 errors, 95 % interval coverage,
bound violations, CPU seconds. Paired Wilcoxon signed-rank tests of the proposed method against every baseline,
Holm correction within (budget, metric); effect = median relative error reduction and rank-biserial correlation.

## Amendments
- 21:40: lfx capped at 800 points (compute time). Validation runs of single-fidelity and hybrid candidates started at 21:30 with LF:HF ratio 1 (they use no extra LF data).
- 23:00: repeats on test fixed for compute time (before any test result): budgets 10/20/40 -> 20 repeats; budgets 80/160 -> 10
  repeats; MFRNP (slowest baseline) -> 10 repeats at 10/20/40 and 5 at 80/160. Main claims use budgets 10/20/40.
- All multi-fidelity and hybrid methods (ours and baselines) use the 128 x 128 grid as the low-fidelity level on test, the
  level selected for the proposed method on val; RICH uses the 64 and 128 solves; LF = raw 128 solve.
- 23:20 (test data still being generated; no test result exists): validation showed that the 95 % intervals of the
  hybrid under-cover (65-87 %). A second, calibrated interval is added for all sklearn-GP-based models (GP, BN-GP,
  AR1, BN-AR1, COR, BN-COR): the predictive standard deviation of the (correction / discrepancy) GP is multiplied by
  s = sqrt(mean z_i^2), where z_i are the closed-form leave-one-out standardised residuals on the training data
  only. Point predictions are unchanged (same Monte Carlo draws); raw and calibrated coverage are both reported.
  Check on val (23:22, BN-COR/BN-AR1/GP/BN-GP, n_H = 10/20/40, 5 repeats; results/calcheck_val.csv): point predictions
  identical to the earlier validation runs; the LOO factor stays close to 1 and changes the coverage by less than
  3 points, i.e. it does not repair the under-coverage. Both intervals are reported; under-coverage is a limitation.

## External data sets (fixed 23:30, before any final external result)
- Natural-fibre RVE data (CC BY 4.0): training subsets of n_g geometry groups (x 7 contrasts) of the 2520-row main set,
  n_g = 6/12/24 (20 repeats) and 48/96 (10 repeats); evaluated once on the authors' independent 60-row FE test set.
  Inputs as in the authors' surrogate (V_f, E_f/E_m, alpha, log10 L, mean inclination, length distribution).
  Only one fidelity exists, so the applicable variant of the method is BN-GP (Voigt/Reuss bounds); compared with GP,
  HT-GP, HT, MLP, XGB, cubic polynomial and the authors' published GP trained on all 2520 rows.
  (An earlier development run on held-out groups of the main set used the waviness level kappa as input; it was
  discarded, the final run uses the authors' monotone inclination variable.)
- Mechanical MNIST (Lejeune 2020; Lejeune & Zhao 2021): HF = UE strain energy, LF = UE-CM-28-perturb; 32 PCA
  features fitted on the 60000 training bitmaps; 10000 test examples. n_H = 100 (10 repeats) and 1000 (3 repeats);
  repeat 0 uses the first n_H training examples, as in the published transfer-learning study. Methods: LF raw,
  GP on HF only (n_H = 100), solver-in-the-loop corrector (log coordinate; no bounds for this quantity).
  Compared with the published CNN and transfer-learning CNN errors computed from the authors' prediction files.

## Frozen selection (00:05, 17.10; test runs started 00:01 with these choices, no test result seen)
Mean over budgets of mape_mean3 on val (repeats: 5 at n_H = 10-80; 2-3 at n_H = 160, the remaining n_H = 160
validation repeats were still running at low priority when the choice was frozen):
BN-COR@128 0.328 | COR@128 0.450 | BN-AR1@128 0.547 | BN-COR@64 0.562 | BN-NARGP@128 0.583 | BN-GP 0.620 |
BN-NARGP@64 0.661 | BN-AR1@64 0.668 | COR@64 0.916 | GP 1.388 | RICH 1.597 | LF@128 2.694 | LF@64 6.543.
Hybrid mode: BN-COR@128 (beats the parametric pick and RICH at every budget). Parametric mode: BN-AR1@128
(6.6 % below BN-NARGP@128, outside the 5 % equivalence margin).
- 00:06: the unfinished n_H = 160 validation repeats were stopped (choice already frozen; memory needed for the test runs).
- 00:20: stream C (MFRNP) started separately at low priority (scripts/run_test_C.sh), same settings as planned.
- 00:47: Mechanical MNIST, after repeat 0 of n_H = 100: the LF quantity (UE-CM-28-perturb, small perturbation) has a
  different magnitude than the HF energy (raw LF MAPE 100 %), so a calibrated LF baseline was added: log-linear map
  log y_H = a + b log y_L fitted on the n_H training pairs. The run was restarted from repeat 0 with this extra
  baseline; nothing else changed (first partial output kept in dev/ext_mnist_first_partial.csv).
- 00:50: ablation added after the first test results of stream A (does not change any selected method): BN-LINCOR =
  the hybrid with the GP correction replaced by a ridge regression on (x, s_L), ridge parameter by LOO on the
  training data; same draws, budgets and repeats (results/test_D.csv).
- Correction (written 04:38; first labelled 04:45 by mistake): the BN-LINCOR entry above is labelled 00:50, but it was written at 00:47:50, i.e. after the
  BN-LINCOR test run had finished (results/test_D.csv, 00:47:48). BN-LINCOR is therefore a post-hoc ablation; the paper
  states this and uses it only for interpretation, not for any selection.
- Correction (04:53): the frozen-selection entry says BN-AR1@128 is '6.6 % below BN-NARGP@128'; correctly, BN-AR1 (0.547)
  is 6.2 % below BN-NARGP (0.583), i.e. BN-NARGP is 6.6 % above BN-AR1. Both readings exceed the 5 % margin, so the
  choice is unchanged.

## Correction of the Halpin-Tsai baseline (10:37, 17.10; found while writing section 1.3 of the paper)
- `micromech.halpin_tsai_set` used the reinforcement factor xi = 1/(4 - 3 nu_m) for G23. The cited source
  (Halpin & Kardos 1976, Polym. Eng. Sci. 16: 344) gives xi = k_m/(k_m + 2 G_m) (plane-strain bulk modulus k_m),
  i.e. 1/(3 - 4 nu_m) for an isotropic matrix. The code was corrected.
- With xi = 1 (G12) and xi = k_m/(k_m + 2 G_m) (G23), the Halpin-Tsai form is algebraically identical to the HS lower
  bound (max. relative difference 4.4e-16 over the design box, results/bounds_stats.json; unit test
  tests/test_models.py::test_halpin_tsai_equals_hs_lower_bound); k23 was already taken as the HS lower value.
  HT therefore coincides with MT and is no longer a separate baseline: 12 existing methods + 3 variants = 15
  comparison methods (was 13 + 3 = 16).
- The 80 HT rows of results/test_A.csv (computed with the wrong factor) were moved to
  results/obsolete/test_A_HT_wrong_xi.csv; scripts/run_test.sh and the default method list no longer contain HT.
- Consequences: Holm families shrink by one (17 instead of 18 tests per budget and metric); at n_H = 10-40 the
  adjusted p of the 20-repeat comparisons becomes 3.2e-5 (was 3.4e-5), at n_H = 80/160 0.033 (was 0.035).
  No other number changes; no selection depended on HT. The statement 'HT G23 below the HS lower bound in all test
  samples' was an artefact of the wrong factor and was removed from the paper.
- Table 4 is now generated by scripts/make_table4.py (same numbers as the hand-built version except the analytic
  row, now Mori-Tanaka; its time is the mean of 1000 calls, results/cost_benchmark.json 'analytic_MT').

## Post-hoc ablation MT-GP (11:02, 17.10; written before the run, motivated by the training-pool analysis of section 2.2)
- Linear 10-fold CV on the training pool (no test data) gives mean MAPE 4.59 % (log y), 1.28 % (log(y/L), L = HS lower
  bound = Mori-Tanaka) and 0.55 % (bound-normalised s). A reviewer may ask whether normalising by the analytical
  estimate alone gives the gain of the bound normalisation. Ablation MT-GP = the GP of the paper on log(y/L)
  (prediction L exp(mu)); same draws, budgets and repeats as GP/BN-GP; appended to results/test_D.csv.
- Interpretation only, no selection. It enters the Holm family like the other ablations (pre-registered rule:
  Holm within budget and metric over all comparisons against BN-COR), i.e. 18 tests per family.
- For the hybrid no separate MT variant is needed: in log(y/L) the correction log(y_H/L) - log(y_L/L) = log(y_H/y_L)
  equals the target of COR.

## Post-hoc ablation BN-COR-X (11:09, 17.10; written before the run, while writing section 2.3)
- Section 2.3 states that the correction GP takes z = (x, s_L) because the coarse solution carries information on
  the particular fibre arrangement. Ablation BN-COR-X: identical to BN-COR (base value s_L of the query kept) but the
  correction GP uses x only, d = d(x). Same draws, budgets and repeats; appended to results/test_D.csv.
  Interpretation only; enters the Holm family like the other ablations (19 tests per family).
- Results (11:20): MT-GP 1.83/0.94/0.61/0.47/0.41 % (BN-GP 1.16/0.67/0.51/0.42/0.39 %); BN-COR-X 0.45/0.38/0.30/0.26/0.21 %,
  not different from BN-COR at any budget (Holm p >= 0.088): the arrangement information enters through the base value
  s_L, the extra input of the correction GP has no measurable effect. The paper states this (sections 2.3, 4.3).
- Holm family now 19 tests per (budget, metric): 20-repeat comparisons p = 3.6e-5; MFRNP (10 repeats) p <= 0.006;
  n_H = 80/160 (10 repeats) p = 0.037. Abstract now quotes the test against the strongest existing method (p < 1e-4).

## Post-hoc robustness analysis: bootstrap over test samples (11:41, 17.10; written before the run)
- The paired Wilcoxon tests capture the variability of the training draw only (fixed test set). To show that the main
  claim does not depend on the particular 300 test RVEs, the per-sample errors of BN-COR and of the strongest existing
  method at each primary budget (BMLP at n_H = 10, 20; MFNN at 40) are recomputed for all 20 repeats with the same
  draws (scripts/bootstrap_test.py; the repeat means must reproduce test_A/test_B), averaged over repeats, and the
  relative error reduction is bootstrapped over the 300 test samples (10000 resamples, percentile 95 % CI).
  No method, setting or number of the evaluation changes; interpretation only.
- Result (12:05): the recomputed repeat means reproduce the evaluation exactly (e.g. BMLP n_H = 10, repeat 0: 1.0723;
  MFNN n_H = 40, repeat 0: 0.6602). Reduction of the mean error vs the strongest existing method (95 % bootstrap CI over
  the test RVEs): 50.0 % (43.5-54.5), 47.7 % (39.1-54.0), 57.6 % (52.4-61.6) at n_H = 10, 20, 40; BN-COR has the lower
  repeat-averaged error on 97.3 %, 96.3 % and 98.0 % of the test RVEs. results/bootstrap_test.json.

## Post-hoc remedy for the hybrid intervals: split-conformal calibration on the validation set (12:06, 17.10; written before the run)
- The hybrid's raw 95 % intervals under-cover on test (60-88 %), the pre-registered LOO scaling does not fix it.
  Standard split-conformal calibration (no refit, no test data): for each repeat, normalised scores |s_H - mu|/sigma
  in the s coordinate are computed on the 100 validation RVEs (never used for fitting), q = the ceil(0.95 (n+1))-th
  smallest score, and the test interval is mu +- q sigma, back-transformed with eq. (11). Per modulus, all budgets
  and repeats as in the evaluation (scripts/conformal.py -> results/conformal.json). Reported as a post-hoc remedy.
- Result (12:15): conformal coverage 96.2/95.8/95.6/96.5/96.4 % (mean of three moduli) at n_H = 10/20/40/80/160, mean
  relative width 6.2/2.9/2.1/1.9/1.5 %. Raw Gaussian intervals in this script: 60.4/66.0/82.7/88.4/88.7 % (the paper's
  raw values 60.0/65.6/82.1/87.9/88.2 % come from the Monte-Carlo intervals of section 2.5). results/conformal.json.

## Extension of the scatter-set analysis to all repeats and budgets (12:15, 17.10; written before the run)
- Section 4.6 used one training draw (n_H = 160, repeat 0) on the 80 scatter RVEs. To give the spread over training
  draws, BN-COR and BN-AR1 are refitted with the draws of all repeats at every budget (same rng as run_budget.py) and
  evaluated on the scatter set; additionally the within-design-point explained variance
  R2_w = 1 - sum(e^2)/sum((y - mean_group(y))^2) measures how well the arrangement-to-arrangement differences are
  resolved (scripts/scatter_eval.py -> results/scatter_eval.json). The scatter set is not the test set.
- Result (13:20): the first run was interrupted by a time limit at n_H = 160, repeat 7; the completed draws were taken
  from its log (3 decimals) and the remaining ones computed by the resumable script. Scatter set, mean over draws:
  hybrid 0.26/0.22/0.15 % (n_H = 160), below the ideal-parametric floor (0.56/0.46/0.27 %) in 10/10 draws at 80 and 160,
  18/20 at 40, 11/20 at 20 and 3/20 at 10; parametric 0.61/0.51/0.30 % (+9.5 to +11.1 % above the floor);
  within-point explained variance of the hybrid 0.83/0.81/0.76 at n_H = 160, 0.36/0.66/0.66 at 40, <= 0.38 at 10-20.

## Application example, figure (13:45, 17.10)
- scripts/application_curve.py refits the parametric mode of scripts/application.py (same draw and seed; design point
  and E2 reproduced to 1e-8 relative) to draw the E2-V_f curve of Fig. 14 and to compute the Mori-Tanaka comparison and
  the cost figures quoted in section 4.9. No new high-fidelity solve; results/application.json unchanged.

## Post-hoc comparison baseline: Voigt-Reuss Net (about 15:37, 17.10; written before any VRN run)
- Reason: pre-submission check; a reviewer can ask for the 2025/2026 bound-guaranteed network itself, not only its
  output-layer idea (BMLP uses the HS bounds and a logit-space loss). Source: Keshav & Fritzen, arXiv:2504.00712
  (spectral normalisation) and IJNME 127 (2026) e70296. Their code is not public, so it is re-implemented.
- Form in this problem: the in-plane (plane strain) and antiplane effective tensors share their eigenprojectors with the
  Voigt and Reuss bounds of the 2D problem, so the spectral normalisation reduces to xi = (V - y)/(V - R) per modulus
  (k23: multiplicity 1, G23: 2, G12: 2; V, R = Voigt/Reuss of the phase moduli at the nominal V_f). Network: tanh MLP,
  sigmoid output = xi, loss = mean over samples of the normalised Frobenius norm sqrt(sum_i m_i (xi_i - xi_hat_i)^2 / 5)
  (their eq. 44) + weight decay; training, ensemble (5), L-BFGS and the 3 x 3 width / weight-decay grid on the
  validation set exactly as for the other networks (scripts/tune_nn.py, repeat-0 draws).
- Evaluation: test set, same draws, budgets and repeats as BMLP (20 repeats at n_H <= 40, 10 at 80/160); appended to
  results/test_D.csv; all tests and tables regenerated with VRN as an existing method (16 comparison methods, Holm over
  20 comparisons). The result is reported whatever it is; if VRN is better than BMLP at some budget, it becomes the
  strongest existing method there and every derived number (abstract, text, tables) is recomputed.
- Correction (15:52, while the VRN test runs were in progress; no change to the runs): the IJNME 2026 paper links public code
  (LGPL-3.0, github.com/DataAnalyticsEngineering/VoigtReussNet); the statement 'their code is not public' above refers to
  the arXiv version only. The implementation here follows the published equations (normalisation eqs. 17-22, loss eq. 25);
  for our 2D problem with shared eigenprojectors it reduces to the per-modulus form; architecture and training are the
  same as for the other networks so that all networks are compared under one training budget.
- Timestamp correction (15:54): the two VRN entries above were first labelled 15:45 and 16:20 by mistake. File times:
  the VRN entry was written before 15:38:12 (backups of results/test_D.csv and nn_hparams.json were taken after it);
  scripts/run_vrn.sh started 15:38:12, VRN tuning on validation finished 15:43:28, the test runs ran 15:43:30-15:52:57
  and were appended to results/test_D.csv at 15:52:57; the code-availability correction was written at 15:52:15.

## Sign error in Hill's nu12 relation (found 16:04, 17.10, in the pre-submission reference check)
- micromech.hill_longitudinal used nu12 = <nu> + a d; the uniform-field argument gives nu12 = <nu> - a d
  (a = (nu_f - nu_m)/(1/k_f - 1/k_m), d = <1/k> - 1/k23); E1 (a^2 term) was correct. Fixed at 16:03 and checked against
  the composite-cylinder closed form for 36 phase/volume-fraction combinations (tests/test_models.py, passes).
- Affected: only the derived constants nu12 and, through E2 = 4 k23 G23/(k23 + m G23) with m = 1 + 4 k23 nu12^2/E1, the
  values of E2 and nu23 (columns mape_E2, mape_nu23, mape_nu12 and max_* in results/test_A-D.csv; the application example
  of section 4.9 and Fig. 14). The moduli G12, G23, k23, all selections, tables 3-4 and tests are unaffected.
- Remedy: scripts/derived_v2.py refits BN-COR and BMLP (the methods whose derived errors are quoted) and MT/LF/RICH with the
  same draws and seeds, checks that every refit reproduces the stored moduli MAPE (< 1e-6 relative), and writes the
  corrected derived errors to results/test_derived_v2.csv; scripts/application.py and application_curve.py are re-run.
  The derived columns of results/test_A-D.csv are kept unchanged as the raw record and are superseded.
- Result (16:16): all refits reproduce the stored moduli MAPE (max relative difference 1.7e-15). Corrected
  derived errors: BN-COR E2 0.39/0.33/0.26 %, nu23 0.36/0.26/0.21 % (was 0.37/0.27/0.22 %), nu12 <= 0.063 %, E1 <= 0.0023 %;
  BMLP E2 0.88/0.75/0.73 % (was 0.87/0.74/0.72 %) at n_H = 10/20/40. BMLP was recomputed for n_H <= 40 only (the quoted
  budgets; the stream for 80/160 was stopped). results/test_derived_v2.csv (merged from _a, _b (10, 20), _c (40)).
- Application example re-run (16:19): with the corrected nu12 the scan selects V_f = 0.530 (was 0.531); the 8 check RVEs are
  generated at this V_f (same seeds), i.e. new arrangements: HF E2/Em = 3.014 +- 0.017 (target +0.46 %, all 8 inside the
  95 % interval 2.95-3.06), hybrid mean/max deviation 0.23/0.44 %, correlation 0.90; Mori-Tanaka 2.76 (-8.4 %).
  The container was restarted at about 15:30 (system uptime), and the CPU times of this run (scan 0.46 s, HF solve median 10.5 s)
  are not comparable with the single-core benchmark of section 4.7 (results/cost_benchmark.json, measured before); 4.9
  therefore quotes the scan time as < 0.5 s and uses the 4.7 benchmark for the HF cost. Old results kept in
  results/obsolete/application*_before_nu12fix.json.

## Structure check of the generated arrangements (16:29, 17.10; pre-submission reference check)
- scripts/order_parameter.py re-runs the generator for the 400 pool RVEs (identical random stream, asserted) and reports
  the global hexatic order |Psi6| (6 nearest neighbours) and the RMS displacement from the initial lattice site.
  Result: |Psi6| median 0.12 (36 uniformly random points: 0.07; initial lattice: 0.96); 0.07-0.12 for area fraction
  V_f(1+g)^2 <= 0.6 with RMS displacement > 1 nearest-neighbour distance; for > 0.6 (38 % of every data set) median 0.28
  and RMS displacement 0.40 spacing, i.e. perturbed lattices. The text (1.1, 4.4, 5) states this as a limitation; the
  earlier wording 'guarantees a disordered structure' was withdrawn. No data or result changes.
- Fig. 5(d) now shows the mean MAPE of the three moduli instead of E2 (the E2 values in test_A-D.csv are superseded by
  results/test_derived_v2.csv, which covers only the methods quoted in the text).

## Pre-submission check of sections 0-3 (written 16:47, 17.10; independent number check of abstract, sections 0-3, Tables 1-2, Figs 1-4)
All numbers matched the result files or the raw data. Statements corrected in the paper (no data or result changed):
- 3.3: the original registration (21:05) had 20 repeats per budget and no primary-budget rule; the repeat counts
  20/10 (MFRNP 10/5) and 'main claims use n_H = 10/20/40' were added at 23:00, after the validation runs had finished
  and before any test result. The paper now says so (it previously dated both to the original registration).
- Abstract: 'pre-registered comparison' reworded to 'comparison following the pre-registered procedure', since VRN was
  added after the test (3.2 says so). 3.2 now discloses that the registered Halpin-Tsai baseline was corrected after the
  test (entry 10:37) and dropped because it then coincides with MT.
- 3.1: the data cost '11 CPU h' was the sum of recorded wall-clock times of 2 parallel workers on a loaded machine; at
  the single-core benchmark rates of 1.2 the five modelling sets cost 2.2 CPU h (scripts/check_sec1_3.py ->
  results/numbers_sec1_3.json, which also stores the other numbers of sections 1-3 that had no result file).
- 3.1 / Table 1: the validation set is also the conformal calibration set of 4.5 (post-hoc, entry 12:06); Table 1
  lists the extra coarse-grid set as LF data of all multi-fidelity methods, not only of the parametric mode.
- 2.5: the LOO-calibrated interval is a test-time amendment (23:20), not part of the original registration.
- Abstract and 5: parametric prediction time 'not more than 0.5 ms' held only for the per-budget medians
  (0.28-0.48 ms; mean at n_H = 80: 0.51 ms; largest single run 0.60 ms); now 'less than 1 ms', as in 2.4.
- 1.1: RMS displacement > 1 nearest-neighbour spacing holds for the bin medians (1.1-4.4), not for every RVE
  (29 of 247 RVEs with area fraction <= 0.6 are below 1); wording changed to the medians.
- 1.4: 7.7 is the median of the per-RVE FE/FFT time ratios (ratio of median times 7.9); wording fixed.
- 2.3 / 4.3: removing s_L from the correction input does not change the mean MAPE significantly, but on G12 at
  n_H = 160 BN-COR is lower in 10/10 repeats (Holm p = 0.039, median reduction 3.8 %); now stated.
- 4.6: the hybrid resolves arrangement differences in all repeats from n_H = 80 on (18/20 at 40); the recommendation
  no longer says 'from 40 on', consistent with the abstract and conclusion (2).
- 3.3: the selection made the hybrid the main method of the paper (the registration had it as a conditional second
  mode); stated in 3.3.
- Fig. 1: the online box now shows sampling of the GP posterior and the sample mean / 95 % interval (as in 2.5) instead
  of the back-transform of the mean; the design variable is labelled lg(E_f/E_m) as in x.
- Clarification of the 'Frozen selection' entry: it was written at 00:05, 4 min after the test runs had been launched
  (00:01) with exactly these choices (scripts/run_test.sh); no test result had been read when it was written.

## Pre-submission check of sections 4.6-4.8, 5 and Figs 11-13 (written 16:50, 17.10)
Independent check against the result files, with exact re-runs of the scatter-set evaluation. Corrections in the paper
(no data or result changed):
- 4.6: hybrid MAPE on the scatter set at n_H = 160 is 0.25 % for G12 (mean 0.2549; the text had 0.26 from rounding the
  3-decimal summary 0.255 a second time; the entry 'Extension of the scatter-set analysis' quoted the same 0.26).
- 4.6 / 5(3): the parametric excess over the scatter floor is quoted as 'about 10 %' instead of '9.5-11.1 %': the
  stored values give +9.5/+10.8/+11.1 %, exact re-runs of the 3-decimal repeats +11.2 % for k23, and repeat 1 at
  n_H = 160 is not bit-reproducible (BN-AR1 k23 MAPE 0.289 on re-run vs 0.285 logged; GP hyper-parameter optimisation
  ends in a slightly different optimum). All other re-run repeats matched the stored values to 3e-7. Note: the floor
  (in-sample mean of the 8 arrangements) is biased low by about sqrt(7/8); this makes the 'below the floor' counts of
  the hybrid conservative.
- 5(3) now names the data set of the n_H = 160 statement (scatter set; the 0.64 % at n_H = 10 is the test set).
- 4.6: the parametric predictions within a design point differ only by Monte Carlo sampling (up to 0.34 %); the text
  now says R2_w is close to or below 0 instead of 'identical predictions, R2_w <= 0'. Fig. 11a error bars (SD over the
  10 repeats) are now explained in the text.
- 4.7 / Table 4: surrogate prediction and training times are single-thread wall-clock times from the test runs; solver
  times are process CPU times from the benchmark. The column is renamed '单次预测时间/s', the note states how each
  column is obtained (median over repeats; training-data time = benchmark solve time x number of solves).
- Fig. 12b: the MFRNP curve had no legend entry; added ('MFRNP (only panel b)').
- Naming: the single-fidelity GP baseline is labelled 'GP' in all figures, tables and the text (figures said 'Kriging').
- 4.8: the natural-fibre surrogate uses the authors' input variables, with the length-distribution type as a binary
  input instead of separate models per type; now stated.
- 1.1: arrangements with area fraction > 0.6 are 38 % of the training pool and 37 % of the test set (the structure-check
  entry said '38 % of every data set'; lfx 39 %, val 38 %, scatter set 50 %); the text now names pool and test set.

## Text revision to the number density and style of JME 2026 articles (20:06-20:41, 17.10; no data, result, figure or table changed)
- Measured against the 424 research articles of JME 2026 (same PyMuPDF extraction for all): numbers in prose 45 per
  1000 zi (above every article) -> 22 (90th percentile); %-numbers in the abstract 8 -> 3, in the conclusion 21 -> 2;
  the numbered contribution list in the introduction (present in 2 % of JME articles) became a paragraph.
- Second pass: results narrated as in JME (figure -> observation -> one number -> cause), 'ben wen' reduced, closing
  application sentence in the abstract, conclusion items start with what was done. Evaluative words only where a
  verified number or test stands behind them.
- Every number of the revised text was checked automatically against the previously verified versions (no new value); each new
  rounding was checked by hand against the result files (e.g. 'about 4.5 %' = 4.46, 'below 90 %' = 88, 'below 35 %'
  = 33.2, 'about 4 times' = 0.84/0.22, 'less than 1/3' = 0.89/2.79, 'at most 1 %' = 0.82, 'about 8 %' = 8.4).
- Cross-references (14 figures, 4 tables, 16 equations, sections) and the 34 references are unchanged; 17 pages.

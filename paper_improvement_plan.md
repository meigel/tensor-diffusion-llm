# Paper improvement plan — Codex 5.5 recommended actions

Priority ordering based on what makes the paper defensible:

## ✅ P0 — Fix the experimental design flaw
~~Current SAT experiment uses ONE random formula per config (all trials share it).
Fix: create fresh domain per trial (formula-seed sweep).
This is the most urgent — a reviewer will flag single-formula results.~~

**Done.** `run_repair_experiment.py` creates a fresh `BoolSatDomain(formula_seed=seed)` per trial.
All SAT results in the paper use n=200 trials with independent formulas.

## ✅ P0 — Add mechanism-isolating baselines
~~Currently we compare "tn_repair vs tn" but that doesn't prove the
repair mechanism itself is what helps. We need:
  - repair + random denoiser
  - repair + local heuristic
These prove: "it's not just remasking until lucky, TN actually helps"~~

**Done.** Both baselines implemented in `make_sudoku_methods()` and `make_sat_methods()`.
Repair + random = 0% across all domains and wrong ratios — the critical control that
proves verifier localization, not remasking itself, is the mechanism.

## ✅ P1 — Increase trials + report confidence
~~n=100 trial results already show the story clearly. We can go to n=200
for the paper table. Add standard error / confidence intervals.~~

**Done.** `run_repair_experiment.py` reports `success_rate ± success_se` (binomial SE).
SAT uses n=200, Sudoku n=100, JSON n=30. Standard errors reported in all tables.

## ✅ P1 — Write the plotting script
~~For the paper we need:
  - success_rate vs wrong_ratio (both domains)
  - steps vs wrong_ratio
  - violation vs wrong_ratio
These make the story visual and much more convincing than tables.~~

**Done.** `plot_results()` in `run_repair_experiment.py` generates PDF plots:
- `sudoku4_success_vs_wrong.pdf`, `sat_success_vs_wrong.pdf`
- `sudoku4_steps_vs_wrong.pdf`, `sat_steps_vs_wrong.pdf`

## ⬜ P1 — Rewrite tex/main.tex
~~Current paper skeleton is Sudoku-centric and treats repair as future work.
The real story is: verifier-guided remasking for robust logical generation.
Rewrite abstract, contributions, methods, and experiments around this.~~

**In progress.** Paper reframed to center denoiser-agnostic remasking (commit 6a2fd8f).
Needs JSON domain + MDLM results integrated.

## P2 — Larger SAT sizes (n=30, n=40)
If contraction remains feasible, add scaling experiments.
If VE slows down, that's still publishable — report the scaling wall.

**Not started.** VTEN contraction may hit exponential blowup for large SAT.
Lowest priority.

---

## Remaining work (ordered by impact)

1. **Definitive JSON experiment** — Compare all 4 denoisers (TN, Local, MLP, MDLM)
   × all 4 policies × 3 wrong ratios (0.0, 0.2, 0.4), n=200.
2. **Fix SAT learned denoiser claim** — Current eval only on formula_seed=0.
   Need 50+ formula sweep before publishing SAT learned denoiser numbers.
3. **Update paper** — Integrate JSON domain, MDLM results, confidence/entropy baselines.
   Fix abstract numbers to match final tables.
4. **Rebuild PDF** — Verify clean build with updated citations.

# Paper improvement plan — Codex 5.5 recommended actions

Priority ordering based on what makes the paper defensible:

## P0 — Fix the experimental design flaw
Current SAT experiment uses ONE random formula per config (all trials share it). 
Fix: create fresh domain per trial (formula-seed sweep).
This is the most urgent — a reviewer will flag single-formula results.

## P0 — Add mechanism-isolating baselines
Currently we compare "tn_repair vs tn" but that doesn't prove the 
repair mechanism itself is what helps. We need:
  - repair + random denoiser
  - repair + local heuristic
These prove: "it's not just remasking until lucky, TN actually helps"

## P1 — Increase trials + report confidence
n=100 trial results already show the story clearly. We can go to n=200
for the paper table. Add standard error / confidence intervals.

## P1 — Write the plotting script
For the paper we need:
  - success_rate vs wrong_ratio (both domains)
  - steps vs wrong_ratio
  - violation vs wrong_ratio
These make the story visual and much more convincing than tables.

## P1 — Rewrite tex/main.tex
Current paper skeleton is Sudoku-centric and treats repair as future work.
The real story is: verifier-guided remasking for robust logical generation.
Rewrite abstract, contributions, methods, and experiments around this.

## P2 — Larger SAT sizes (n=30, n=40)
If contraction remains feasible, add scaling experiments.
If VE slows down, that's still publishable — report the scaling wall.

## Implementation order
1. Fix SAT domain to seed formula generation
2. Add repair+random, repair+local baselines to both domain configs
3. Run definitive experiment (both domains, n=100, all baselines)
4. Generate paper plots
5. Rewrite paper skeleton

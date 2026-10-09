# Pre-submission checks (Oct 8). Reproduce: PYTHONPATH=. python scripts/review_checks.py ; PYTHONPATH=. python scripts/gcr_rescore.py

## % of the oracle F1 gain recovered by learned:dev @2% FPR (paired bootstrap over questions, 95% CI)
- WebQSP-1328: 78.7% [72.4, 85.1]   - CWQ-3231: 72.8% [64.5, 81.9]   - WebQSP-500 seed 4: 81.7% [70.3, 93.6]

## WebQSP -> CWQ leakage check (detector trained on WebQSP dev = test ids 1000:1300)
- CWQ test questions with the same WebQSP seed id as a dev question: 203/3531; sharing a topic entity with any dev question: 618/3531.
- F1 gain of learned vs none: all 3231 q +8.3 [+6.8, +9.7]; non-overlapping 2591 q +7.9 [+6.4, +9.5]; overlapping 640 q +9.8 [+6.4, +13.1]. Transfer claim holds without the overlap.

## Attack strength vs the original (RoG, WebQSP-500, 4-bit): relative drop clean -> attacked
- Ours: Hit 83.2 -> 78.0 (-6%), F1 67.7 -> 45.9 (-32%).  Zhao et al. (arXiv 2507.08862, Table, from search snippets, to verify): EM 45.76 -> 8.85 (-81%), Hits@1 79.61 -> 48.46 (-39%), Hit 85.87 -> 75.61 (-12%).
- Our attack is weaker than the original on every metric we can compare. State this; consider a stronger-budget run (K=8 / N=10).

## GCR scoring (400 q): the low clean F1 comes from taking the union of ~8 distinct answer strings over 10 beams
- first answer only: clean hit 78.5 F1 50.5 precision 87.5 | first 3: F1 49.8 | all 10: hit 85.2 F1 35.7 precision 39.0.
- Attack effect on GCR under each aggregation: F1 diff first-1 -2.2 [-3.8,-1.0], first-3 -1.5 [-2.4,-0.7], all-10 -0.9 [-1.3,-0.4]; planted answer top-1 2.5%. "GCR barely affected" holds under all of them.
- Clean F1 50.5 / hit 78.5 is still below the published GCR figure (4-bit weights, plain beam search); bf16 rerun with a fixed answer aggregation is still needed.

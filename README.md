# Defending KG-RAG against knowledge poisoning

Code, run files and verification scripts for the paper

> Manish Nandish, Rajiv Misra, Midhunchakkaravarthy Janarthanan.
> *Defending KG-RAG Against Knowledge Poisoning: Calibrated Path Filtering and Its Limits Under Adaptive Attack.* (under review)

The paper studies a defence for knowledge-graph retrieval-augmented generation (RoG, GCR on WebQSP and CWQ) against the triple-insertion attack of Zhao et al. (arXiv 2507.08862): a gradient-boosted path detector, a conformal-risk-control removal threshold, and a question-level gate.

## Layout

| Path | Contents |
|---|---|
| `kgrag/` | attack re-implementation and variants (`attack.py`), RoG and GCR runners, path features and detector (`pathdata.py`, `learned.py`), conformal threshold (`conformal.py`), gate (`gate.py`), baselines (`defend.py`, `ragdefender.py`, `kge.py`), the end-to-end runner (`defended_run.py`), table builder (`report.py`) and unit tests (`test_*.py`) |
| `scripts/` | worst-case tables, GCR defence report and decomposition, conformal validity simulation, figure generation |
| `runs/kg/`, `runs/kg_strict/`, `runs/kg_mg/` | per-question result files: original poisoned-path labels, strict labels, MultiGraph ablation. One JSON line per question: `id`, `prediction`, `hit`, `f1`, `precision`, `recall`, `a_precision`, `a_hit1`, `a_mrr`, `n_paths`, `n_kept` |
| `review/` | the verification scripts and `claims_ledger.csv`, which lists every number in the paper next to the value regenerated from the run files |
| `paper/` | LaTeX source and figures |

## Reproducing the numbers

Tables are recomputed from the saved predictions on a CPU; no GPU is needed to check them.

```bash
pip install -r requirements-hpc.txt
python -m kgrag.report                 # results table with bootstrap intervals (runs/kg)
KG_RUNS=runs/kg_strict python -m kgrag.report
python scripts/worst_case_tables.py    # worst-case gains over the attacks
KG_RUNS=runs/kg_strict PYTHONPATH=. python scripts/gcr_defence_report.py
python review/ledger_t1_6.py           # paper vs regenerated values, Tables 1-3
python review/ledger_t4_13.py          # Tables 4-13
```

The `KG_RUNS` environment variable selects the run folder. Re-running the language models needs the public benchmarks (`rmanluo/RoG-webqsp`, `rmanluo/RoG-cwq`), the RoG and GCR checkpoints, and Qwen2.5-7B-Instruct; the generators in `kgrag/attack.py` rebuild the poisoned graphs. The 4-bit runs used a 16 GB GPU (transformers 4.57.6).

The large path files (`paths_*.jsonl`, about 4,300 triples per question) are not included; `python -m kgrag.pathdata` regenerates them, and some of the verification scripts need them.

## Labels

A retrieved path is *poisoned (strict)* only if it uses an inserted triple that is not already in the clean graph in either orientation. Which rows use strict labels and which keep the original labels is stated in Section 5 of the paper and in every table caption.

## License

MIT (see `LICENSE`). The benchmarks and models keep their own licenses.

## Contact

Manish Nandish, IIT Patna: manish_25s21res58@iitp.ac.in

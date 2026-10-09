# GCR pilot: clean vs attacked graph (WebQSP test questions 0:400; 4-bit weights, plain beam search k=10)
Deviation from the reference: group beam search needs trust_remote_code (not enabled). Clean F1 is far below the published GCR number, so this is NOT yet a faithful reproduction (see HPC_AGENT_PROMPT.md, T0b).

| | Hit | F1 | Precision | planted answer top-1 |
|---|---|---|---|---|
| clean | 85.2 [81.5, 88.8] | 35.7 [33.3, 38.2] | 39.0 | - |
| attacked | 84.5 [81.0, 88.0] | 34.9 [32.3, 37.4] | 38.3 | 2.5 [1.0, 4.0] |
| paired difference | -0.8 [-2.0, +0.2] | -0.9 [-1.3, -0.4] | -0.7 | |
RoG on the same attack: planted answer top-1 about 30%.

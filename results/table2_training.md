# Table 2 - Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | 2.5076 | val. loss: 2.445 | 71 min / Tesla T4 |
| Reward model | 0.5926 | pairwise acc.: 63.1% | 103 min / Tesla T4 |
| PPO | 0.1364 | final mean r_phi: 0.971 | 50 min / Tesla T4 |
| DPO | 0.6267 | implicit reward acc.: 61.1% | 186 min / Tesla T4 |

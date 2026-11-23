# Table 4 - Human evaluation (30 posts)

| | SFT picked | PPO picked | DPO picked |
|---|---|---|---|
| Amama | 16 | 1 | 13 |
| Zeenat (round 2) | 7 | 7 | 16 |
| **Total (60 votes)** | **23** | **8** | **29** |

Agreement between us: **19 / 30 (63%)** (chance is about 33%).

Round 1 (Zeenat): SFT 10, PPO 9, DPO 11; agreement 11 / 30 (37%).

**Why two rounds.** In round 1 Zeenat's sheet showed Amama's picks and Zeenat never picked B, so Zeenat repeated the
picks on a clean, reshuffled sheet (positions A/B/C = 10/12/8). The decision to repeat was made after seeing the round 1
result, and Zeenat had already seen the posts; Zeenat chose the same model in both rounds on 17 / 30 posts. Amama was not
repeated (positions 19/7/4). With 30 posts, treat this as a small, noisy check. In `human_eval.csv`, Zeenat's round 2
picks are translated to the original letters.

#!/usr/bin/env bash
# Runs the whole pipeline top to bottom on one GPU (Colab/Kaggle T4).
set -e
python starter/check_starter.py | tee results/check_starter.txt
python task1_data.py
python train_sft.py
python train_reward.py
python train_ppo.py
python train_dpo.py
python evaluate.py
python plots.py
python human_eval.py make
echo "Now fill results/human_eval.csv (both members), then: python human_eval.py score"

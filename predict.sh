#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -w gpu2
#SBATCH -p gpu


python -m ares.predict /home/zhangyi/3dRNA/ARES/ares_release/article_data/dataset/TestII/decoys/1EZN /home/zhangyi/3dRNA/ARES/ares_release/article_data/Model/epoch50/Model_6/model6.ckpt 1.csv -f pdb --nolabels --gpus=1 --num_workers=8 

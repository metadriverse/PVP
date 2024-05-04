#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-last_ratio=0.1-remove_loss_3=True \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=False \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=True \
    --last_ratio=0.1 \
    > "0503-exp1-seed${seeds[$i]}.log" 2>&1 &
done


# Loop over each GPU
for i in {2..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-last_ratio=0.1-remove_loss_1=True \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=True \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --last_ratio=0.1 \
    > "0503-exp2-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {4..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-last_ratio=0.1-allloss \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=False \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --last_ratio=0.1 \
    > "0503-exp3-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {6..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-last_ratio=0.7-allloss \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.25 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=False \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --last_ratio=0.7 \
    > "0503-exp4-seed${seeds[$i]}.log" 2>&1 &
done

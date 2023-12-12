#!/bin/bash

# 设置 Python 解释器路径和脚本路径
PYTHON=/home/lbk/ENTER/envs/llmpde/bin/python
SCRIPT=/home/lbk/llm/llama4pde/train.py

# 调用 Python 脚本并传递参数
accelerate launch --num_processes 2 --main_process_port 29505 $SCRIPT \
    --gpu 0 7   \
    --seed 1234 \
    --S0 13 \
    --S1 1440 \
    --S2 721\
    --downsample 2 \
    --epochs 30 \
    --batch_size 1 \
    --accumulation_steps 2 \
    --ntrain 4018 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --scheduler_type "steplr" \
    --scheduler_step 4018 \
    --scheduler_gamma 0.5 \
    --train_object "mae" \
    --lora_r_llama 8 \
    --lora_alpha_llama 16 \
    --train_auto \
    --dim 64 \
    --final_dim 4096 \
    --surface_in_chans 4 \
    --upper_in_chans 5 \
    --if_surface_const_mask True \
    --act_type "newgelu" \
    --norm_type "ln" \

    
    

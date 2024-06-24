#!/bin/bash


default_prompt="na"

# change model to generate heatmap for different models
for num_docs in 3 9
    do
    for model in "tulu-2-7b_formated"
        do
        python src/acc_heatmap_untuned_var_model.py \
            --num-docs $num_docs \
            --model $model \
            --default-prompt $default_prompt 
        done
    done

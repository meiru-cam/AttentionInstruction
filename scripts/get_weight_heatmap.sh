#!/bin/bash

num_docs=3
for model_name_or_path in "Mistral-7B-Instruct-v0.2"; do
    for attention_level in token; do
            python src/visualize_attention_weight.py --attention_level $attention_level --num_doc $num_docs --model $model_name_or_path --have_docid
        done
    done

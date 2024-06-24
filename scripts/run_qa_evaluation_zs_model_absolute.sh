#!/bin/bash

# this is for RQ2

# run with different number of documents and models
# with 3-documents and 9-documents
# inference other models 
# "Llama-2-7b-chat-hf" "Meta-Llama-3-8B-Instruct" "Mistral-7B-Instruct-v0.2" "Mistral-7B-Instruct-v0.1"

for num_docs in 3 9
    do
    for model_name in "tulu-2-7b"
        do
        python -u src/get_qa_responses.py \
            --num-docs $num_docs \
            --max-new-tokens 100 \
            --num-gpus 1 \
            --model $model_name \
            --attention-level token \
            --have-docid
        done
    done

for num_docs in 3 9
    do
    for model_name in "tulu-2-7b"
        do
        python -u src/get_qa_responses.py \
            --num-docs $num_docs \
            --max-new-tokens 100 \
            --num-gpus 1 \
            --model $model_name \
            --attention-level token \
            --reverse-idx
        done
    done

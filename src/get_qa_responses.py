#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run language models to get responses.

Currently supports:

- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Meta-Llama-3-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mistral-7B-Instruct-v0.2
- allenai/tulu-2-7b

The retrieval results are used in the exact order that they're given.
"""
import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

import os
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen

from utils.prompting_new import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    # Numpy module
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")
    else:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_model(model_name, num_gpus, max_prompt_length, hf_cache_path, seed):
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    
    logger.info("Loading model")
    memory_limit = 0.85

    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        download_dir=hf_cache_path,
        # load_format="pt",
        max_num_batched_tokens=max_prompt_length,
        seed=seed,
        gpu_memory_utilization=memory_limit
    )
    return model, tokenizer


def main(
    input_path,
    model_name,
    model,
    tokenizer,
    temperature,
    top_p,
    closedbook,
    attention_to,
    attention_level,
    have_docid,
    replace_docid,
    reverse_idx,
    prompt_mention_random_ordering,
    use_random_ordering,
    max_new_tokens,
    max_prompt_length,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False

    inputs_all = []
    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            inputs_all.append(input_example)

    if args.fast_eval:
        inputs_all = np.random.choice(inputs_all, 500, replace=False)
    
    for input_example in inputs_all:
        # Get the prediction for the input example
        question = input_example["question"]
        if closedbook:
            documents = []
        else:
            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            if not documents:
                raise ValueError(f"Did not find any documents for example: {input_example}")

        if use_random_ordering:
            # Randomly order only the distractors (isgold is False), keeping isgold documents
            # at their existing index.
            (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
            original_gold_document = documents[original_gold_index]
            distractors = [doc for doc in documents if doc.isgold is False]
            random.shuffle(distractors)
            distractors.insert(original_gold_index, original_gold_document)
            documents = distractors

        if closedbook:
            prompt = get_closedbook_qa_prompt(question)
        else:
            prompt = get_qa_prompt(
                question,
                documents,
                attention_to,
                attention_level,
                have_docid,
                replace_docid,
                reverse_idx
            )

        if "tulu" in model_name:
            prompt = format_tulu(prompt)

        prompt_length = len(tokenizer(prompt)["input_ids"])
        if max_prompt_length < prompt_length:
            print("need to truncate prompt")
            logger.info(
                f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                f"is greater than maximum prompt length {max_prompt_length}"
            )
            continue

        prompts.append(prompt)
        examples.append(deepcopy(input_example))
        all_model_documents.append(documents)

    logger.info(f"Loaded {len(prompts)} prompts to process")

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)


    raw_responses = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in raw_responses]

    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def format_tulu(message: str):
    lines=["<|user|>", message, "<|assistant|>"]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", help="Number of documents", type=int, required=True)
    parser.add_argument("--seed", help="Random seed to use", type=int, default=0)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument("--attention-level", help="Whether to focus on a token or position", type=str, default="token")
    parser.add_argument("--have-docid", action="store_true", help="Whether the documents have a document ID")
    parser.add_argument("--pos-placeholder", action="store_true", help="Whether to add position placeholders to the prompt")
    parser.add_argument("--replace-docid", action="store_true", help="Whether to replace the document ID in the prompt")
    parser.add_argument("--nonexist-idx", action="store_true", help="Whether to shuffle the document index")
    parser.add_argument("--reverse-idx", action="store_true", help="Whether to reverse the document index")
    parser.add_argument("--controled-shuffle", action="store_true", help="shuffle the index but keep the order of gold documents")
    parser.add_argument("--use-fst", action="store_true", help="Use first|second|third as the attention part")
    parser.add_argument("--multi-doc", action="store_true", help="pay attention to multiple documents")

    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--hf-cache-path", help="Path to huggingface cache to use.")
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=4096,
    )
    parser.add_argument("--fast-eval", help="for faster check of the result", default=False, action="store_true")
    args = parser.parse_args()

    print(args)

    logger.info("running %s", " ".join(sys.argv))

    model_name = ""

    max_prompt_length = 4096
    if "Llama" in args.model:
        model_name = "meta-llama/" + args.model
        if "Llama-3" in args.model:
            max_prompt_length = 8192
    elif "Mistral" in args.model:
        model_name = "mistralai/" + args.model
        max_prompt_length = 32768
    elif "tulu" in args.model:
        model_name = "allenai/{}".format(args.model)
        max_prompt_length = 8192
    

    interval = args.num_docs // 3
    start_idx = interval//2
    gold_indexs = [start_idx, start_idx + interval, start_idx + 2*interval]

    
    if args.attention_level == "token":
        ref="idx"
        if args.replace_docid:
            ref="posword"
        attention_parts = [str(gold_id+1) for gold_id in gold_indexs]
        attention_parts.append("na")
    elif args.attention_level == "position":
        attention_parts = ["beginning", "midsection", "tail", "all", "na"]

    print("gold_indexs", gold_indexs)
    print("attention_parts", attention_parts)

    set_seed(args.seed)
    model, tokenizer = init_model(model_name, args.num_gpus, max_prompt_length, args.hf_cache_path, args.seed)

    for gold_idx in gold_indexs:
        gold_at = gold_idx+1
        input_path = f"data/nq_data/total_{args.num_docs}_documents_reverse/nq-open-{args.num_docs}_total_documents_gold_at_{gold_at}_with_least_distractors.jsonl.gz"
        # Run the model for each attention part
        for attention_to in attention_parts:
            # logger.info("start running %s", sys.argv[0], "gold at", gold_at, "attention to", attention_to)
            print("start running ", "gold at", gold_at, "attention to", attention_to)

            exp_setting = []
            
            exp_setting.append("have_docid" if args.have_docid else "no_docid")
            exp_setting.append(f"with_reversed_{ref}" if args.reverse_idx else "")
            exp_setting.append(f"with_ascending_{ref}" if args.replace_docid else "")

            exp_setting = "_".join([setting for setting in exp_setting if setting])
            print(exp_setting)

            if "tulu" in args.model:
                output_path = f"results/zero_shot/qa_predictions/{args.attention_level}_level_{exp_setting}/nq_total_{args.num_docs}_documents_reverse/{args.model}_formated/gold_at_{gold_at}_attention_to_{attention_to}.jsonl.gz"
            else:
                output_path = f"results/zero_shot/qa_predictions/{args.attention_level}_level_{exp_setting}/nq_total_{args.num_docs}_documents_reverse/{args.model}/gold_at_{gold_at}_attention_to_{attention_to}.jsonl.gz"
            
            if os.path.exists(output_path):
                print("output already exists, skip")
                continue

            main(
                input_path,
                model_name,
                model,
                tokenizer,        
                args.temperature,
                args.top_p,
                args.closedbook,
                attention_to,
                args.attention_level,
                args.have_docid,
                args.replace_docid,
                args.reverse_idx,
                args.prompt_mention_random_ordering,
                args.use_random_ordering,
                args.max_new_tokens,
                max_prompt_length,
                output_path,
            )

            # logger.info("finished running %s", sys.argv[0], "gold at", gold_at, "attention to", attention_to)
            print("finish running ", "gold at", gold_at, "attention to", attention_to)

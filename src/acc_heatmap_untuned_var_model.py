#!/usr/bin/env python3
import string
from typing import List

import regex

import argparse
import json
import logging
import statistics
import sys
from copy import deepcopy

import os
from glob import glob
from tqdm import tqdm
from xopen import xopen

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pprint import pprint
from icecream import ic

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

METRICS = [
    (best_subspan_em, "best_subspan_em"),
]

def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


def custom_sort_key(s, pos_keys):
    """Custom sort function to sort first by gold prompt position, then by position prompt"""
    number = int(s.split("/")[-1].split("_")[2])

    order = {pos_keys[0]: 1, pos_keys[1]: 2, pos_keys[2]: 3}
    re_expression = "(" + "|".join(pos_keys) + ")"
    position = re.search(re_expression, s).group()

    return (number, order[position])


def get_heatmap_score(prediction_files, pos_prompt, attention_level):
    """Get heatmap scores for a list of prediction files

    Args:
        prediction_files (list): list of prediction files to compute scores for
        pos_prompt (str): position prompt to filter prediction files

    Returns:
        list: list of scores for each prediction file
    """

    if attention_level == "position":
        pos_keys = pos_prompt
    elif attention_level == "token":
        pos_keys = [key+"." for key in pos_prompt]

    if pos_prompt != "na":
        prediction_files = [f for f in prediction_files if any(pos_key in f for pos_key in pos_keys)]
        prediction_files = sorted(prediction_files, key=lambda s: custom_sort_key(s, pos_prompt))
    else:
        prediction_files = sorted(prediction_files, key=lambda s: int(s.split("/")[-1].split("_")[2]))

    all_scores = []
    for input_path in prediction_files:
        all_examples = []
        with xopen(input_path) as fin:
            for line in tqdm(fin):
                input_example = json.loads(line)
                all_examples.append(input_example)
        
        all_example_metrics = []
        for _, example in enumerate(all_examples):
            all_example_metrics.append(get_metrics_for_example(example))

        # Average metrics across examples
        for (_, metric_name) in METRICS:
            average_metric_value = statistics.mean(
                example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
            )        
        all_scores.append(average_metric_value*100)

    return all_scores


def plot_heatmap_multi(all_scores, default_score, save_dir, title, gold_indexs, attention_parts, attention_level):
    """Plot heatmap for single/multiple attention prompts (inclusive of default score comparison)
    The plot should be 4x1 subplots with each subplot representing a different setting

    Args:
        all_scores (list): list of scores for each gold position and prompt position combination
        default_score (list): list of scores for files with default prompt over gold position
        args (dict): arguments for plotting
    """

    # Number of subplots
    num_plots = len(all_scores)

    if len(all_scores) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(11, 4), squeeze=False)
    elif len(all_scores) == 6:
        fig, axes = plt.subplots(2, 3, figsize=(7.5, 6), squeeze=False)
    axes = axes.flatten()

    ic(attention_parts)
    ic(all_scores)

    # get limits
    vmin = min([min(scores) for _, scores in all_scores])
    vmax = max([max(scores) for _, scores in all_scores])
    
    for idx, (model, scores) in enumerate(all_scores):
        data = np.reshape(scores, (3, 3))

        score_default = np.reshape(default_score[idx], (3, 1))
        data_diff = data - score_default
        
        # Generate custom annotations combining original data and differences
        annot = np.vectorize(lambda x, y: f"{x:.2f}\n+{y:.2f}" if y >= 0 else f"{x:.2f}\n{y:.2f}")(data, data_diff)

        ic(data)
        ic(annot)

        # Plot heatmap using data_diff for color scale but custom annotations for display
        if idx in [0, 3]:
            mapcolor = "Reds"
        if idx in [1, 4]:
            mapcolor = "Blues"
        if idx in [2, 5]:
            mapcolor = "Greens"
        

        sns.heatmap(data, annot=annot, cmap=mapcolor, ax=axes[idx], cbar=False, fmt='', vmin=max(scores)-15)

        axes[idx].set_title(model)

        if idx == 0 or (idx==3 and len(all_scores)==6):
            axes[idx].set_ylabel('Gold Document Position')
            axes[idx].set_yticklabels(gold_indexs)
        else:
            axes[idx].set_yticks([])

        axes[idx].set_xticklabels(attention_parts[idx], rotation=30)
    

    # set a overall title
    # fig.suptitle(title)
    # fig.supxlabel('Prompt Document Position')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_dir + title + ".png")

    # Display the heatmaps
    plt.show()


def get_prediction_files(model, prediction_path):
    if "chatgpt" in model or "gpt4" in model:
        return glob(f"{prediction_path}*.jsonl")
    else:
        return glob(f"{prediction_path}*.jsonl.gz")


def filter_files(files, default_prompt):
    na_files = [f for f in files if default_prompt in f]
    att_files = [f for f in files if "na" not in f and "random" not in f]
    return na_files, att_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", help="num_docs", type=int, required=True)
    # parser.add_argument("--att-prompt", help="Which prompt to use", type=str, default="9")
    parser.add_argument("--default-prompt", help="random or na", type=str, default="")
    parser.add_argument("--model", help="which model to plot", type=str, required=True)

    args = parser.parse_args()

    # Set up save directory
    plot_save_dir = f"results/zero_shot/plots/acc_heatmap/"
    os.makedirs(os.path.join(plot_save_dir), exist_ok=True)


    # Load position dictionary
    pos_dict = json.load(open("src/utils/prompts/pos_dict.json", "r"))


    settings = ["position_level_no_docid", "token_level_have_docid", 
                "token_level_have_docid_with_ascending_posword"]
    subtitles = ["relative attention with docid", "absolute attention with docid", 
                 "absolute attention with position"]

    # at finetuning stage, we selected prompt 9, no need to have multiple prompts
    prompt_index = 9
    all_att_scores = []
    all_na_scores = []
    all_att_parts = []
    # for prompt_index in att_prompts:

    interval = args.num_docs // 3
    start_idx = interval//2
    gold_indexs = [start_idx, start_idx + interval, start_idx + 2*interval]

    gold_ticks = [f"Doc {gold_id+1}" for gold_id in gold_indexs]

    title = f"{args.num_docs}_documents_{args.model}"

    for (exp_setting, subtitle) in zip(settings, subtitles):
        if "token" in exp_setting:
            attention_level = "token"
            attention_parts = [str(gold_id+1) for gold_id in gold_indexs]
        elif "position" in exp_setting:
            attention_level = "position"
            attention_parts = ["beginning", "midsection", "tail"]

        prediction_dir = f"results/zero_shot/qa_predictions/{exp_setting}/nq_total_{args.num_docs}_documents_reverse/{args.model}/"
        prediction_files = get_prediction_files(args.model, prediction_dir)

        na_files, att_files = filter_files(prediction_files, args.default_prompt)
        att_scores = get_heatmap_score(att_files, attention_parts, attention_level)
        all_att_scores.append((subtitle, att_scores))
        na_scores = get_heatmap_score(na_files, args.default_prompt, attention_level)
        all_na_scores.append(na_scores)
        all_att_parts.append(attention_parts)
    plot_heatmap_multi(all_att_scores, all_na_scores, plot_save_dir, title, gold_ticks, all_att_parts, attention_level)

# for loading the model
from transformers import  LlamaForCausalLM, LlamaTokenizer, TrainingArguments, set_seed, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# for prompting
from utils.prompting_new import Document, get_qa_prompt

# for general use
import torch
import os
import numpy as np
from tqdm import tqdm
import json
import argparse
from xopen import xopen
from copy import deepcopy

# for visualize
# from bertviz import head_view
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# for warnings
import sys
import logging

# logging.set_verbosity_error()
logging.basicConfig(stream=sys.stdout)
# logging.verbose = 1

# for debug
from icecream import ic
ic.disable()  # Disable ic() calls


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_argument():
    parser = argparse.ArgumentParser(description="Visualize the attention weights of a model")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="the base model name or path")
    parser.add_argument("--num_docs", type=int, default=3, help="Number of documents in the context")
    parser.add_argument("--example_idx", type=int, default=17, help="Which example to visualize")
    parser.add_argument("--posatt_prompt", type=int, default=9, help="Which attention prompt to use")
    parser.add_argument("--attention_level", type=str, default="token", help="Position level to visualize")
    parser.add_argument("--have_docid", action="store_true", help="Whether to include document id in the prompt")
    return parser.parse_args()


def load_inputs(input_path):
    inputs_doc = []
    total_gold_length = 0
    total_distractor_length = 0
    total_n = 0
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            for idx, doc in enumerate(input_example["ctxs"]):
                if doc["hasanswer"] is False:
                    total_distractor_length += len(doc["text"].split(" "))
                else:
                    total_gold_length += len(doc["text"].split(" "))
            inputs_doc.append(input_example)
            total_n += 1
    avg_gold_length = total_gold_length / total_n
    avg_distractor_length = total_distractor_length / (total_n*2)
    print(f"Average gold document length: {avg_gold_length}")
    print(f"Average distractor document length: {avg_distractor_length}")

    return inputs_doc


def format_chat_prompt(message: str):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
    )
    lines = ["<s>[INST] <<SYS>>", DEFAULT_SYSTEM_PROMPT, "<</SYS>>", "", f"{message} [/INST]"]
    return "\n".join(lines)


def separate_question_and_context(input_example):
    question = input_example['question']
    documents = []
    for ctx in deepcopy(input_example["ctxs"]):
        documents.append(Document.from_dict(ctx))
    if not documents:
        raise ValueError(f"Did not find any documents for example: {input_example}")
    return question, documents


def sent_split(token_ids, tokenizer):
    # get the token ids of newline
    idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("\n"))
    
    indices = [i for i, x in enumerate(token_ids) if x == idx[1]] # 13 is the token id for newline
    # print(indices)

    indices.insert(0, 0)
    indices.append(len(token_ids))

    split_sents = []
    split_ranges = [(indices[i], indices[i+1]) for i in range(len(indices)-1) if indices[i+1] - indices[i] > 1]

    for i, (start, end) in enumerate(split_ranges):
        split_sents.append(token_ids[start:end])
    
    # for i in range(len(split_sents)):
    #     print(tokenizer.decode(split_sents[i]).split()[0])
    #     print("--"*20)
    
    return split_sents, split_ranges


def get_attention(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    inputs = inputs.to('cuda')  # Move input tensors to GPU

    outputs = model(inputs)
    attention = outputs[-1]  # Output includes attention weights when output_attentions=True
    return attention


## change attention mask with 0 and 1, changing over each 100 chunk
def get_attention_modified_mask(prompt, model, tokenizer):
    inputs = tokenizer.encode_plus(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    seq_length = input_ids.size(1)
    attention_mask = torch.ones(1, seq_length, dtype=torch.float32)  # Initialize with all ones
    chunk_size = 100

    for i in range(0, seq_length, chunk_size):
        end_idx = min(i + chunk_size, seq_length)
        chunk_length = end_idx - i
        chunk_value = 1 / (i // chunk_size + 1)  # Calculate the value for the current chunk
        attention_mask[0, i:end_idx] = chunk_value

    attention_mask[0,100:200] = 2.0

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    attention_weights = outputs.attentions  # Output includes attention weights

    return attention_weights


def min_max_normalization(X, axis):
    """
    Min-Max Normalization by Row

    Parameters:
    X (numpy.ndarray): Input data to be normalized

    Returns:
    numpy.ndarray: Normalized data
    """
    X_norm = X.copy()
    X_max = np.max(X, axis=axis, keepdims=True)
    X_min = np.min(X, axis=axis, keepdims=True)

    X_norm = (X_norm - X_min) / (X_max - X_min + 1e-8)

    return X_norm


def attention4d_to_2d(attention_ori, split_ranges, selected_parts):
    """
    Convert 4D attention tensor to 2D attention matrix

    Parameters:
    attention (torch.Tensor): 4D attention tensor

    Returns:
    numpy.ndarray: 2D attention matrix
    """

    # Convert tensors to numpy arrays
    attention_ori = [head.cpu().detach().numpy().squeeze() for head in attention_ori]
    attention_ori = np.array(attention_ori)

    # average over heads
    attention = np.mean(attention_ori, axis=0) # (32, len(token_ids), len(token_ids)

    # select the last token
    attention = attention[:, -1, :] # (32, len(token_ids)) 
    # attention = attention[:,:,-1]

    # normalize attention scores over each layer
    # attention = min_max_normalization(attention, 1) # (32, len(token_ids))

    # swap the last two dimensions
    attention = np.swapaxes(attention, 0, 1) # (len(token_ids), 32)

    ic(split_ranges)
    # print(attention.shape)
    # get average attention score for different sent split in each layer
    split_atten = [] # (9, 32) 

    ic(selected_parts)
    
    for i in selected_parts:
        sub = attention[split_ranges[i][0]:split_ranges[i][1]]
        score_total = np.mean(sub, axis=0)
        split_atten.append(score_total)

    # Print shapes for verification
    # print("Shape of original attention", np.array(attention).shape)  # (32, 20)
    # print("Shape of splitted attention", np.array(split_atten).shape)  # (4, 32)
    
    return attention, split_atten


def load_model(model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path) #could set use_fast=False
    tokenizer.pad_token = tokenizer.eos_token # set the pad token to be the eos token
    tokenizer.padding_side = "left" # set the padding side to be left

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager", #LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`.
        # local_files_only=True,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
        # load_in_8bit_fp32_cpu_offload=True,
        device_map="auto",
        output_attentions=True,
    )
    
    if adapter_path == None:
        return model, tokenizer
    tuned_model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

    # # if merge two adapters
    # model.add_weighted_adapter(
    #         adapters=['lora-1', 'lora-2'],
    #         weights=[0.5, 0.5],
    #         adapter_name="combined",
    #         combination_type="svd",
    #     )
    return tuned_model, tokenizer


def plot_weights_subplots(split_attn_list, split_sents_list, titles, save_to, fig_name, fig_title, ncols=3):
    nrows = len(split_attn_list) // ncols + (len(split_attn_list) % ncols != 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))

    # Calculate vmax for each row, so that the same colorbar scale is used for each row, same prompt has the same colorbar scale
    row_vmax = [0, 0, 0]
    for i in range(nrows):
        r_max = 0
        for j in range(ncols):
            if i*ncols+j >= len(split_attn_list):
                break
            split_attn = split_attn_list[i*ncols+j]
            r_max = max(r_max, np.max(split_attn))
        row_vmax[i] = r_max
        

    for i, (split_attn, split_sents, title) in enumerate(zip(split_attn_list, split_sents_list, titles)):
        row, col = i // ncols, i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        words = split_sents
        attn_values = np.array(split_attn)

        x_stick = np.linspace(0, 32, 9)  # 9 sticks

        im = ax.imshow(attn_values, cmap='Reds', aspect='auto', vmin=0, vmax=row_vmax[row])

        # Set x-ticks and labels
        if col == 0:  # Show x-ticks and labels only on the leftmost subplots
            ax.set_yticks(np.arange(len(words)))
            ax.set_yticklabels(words)
            ax.set_ylabel("Input Segments", fontsize=12)
            plt.setp(ax.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_yticks([])

        # Set y-ticks and labels
        if row == nrows - 1:  # Show y-ticks and labels only on the bottom subplots
            ax.set_xticks(x_stick)
            ax.set_xticklabels([str(int(i)) for i in x_stick])
            ax.set_xlabel("Layer Ids", fontsize=12)
        else:
            ax.set_xticks([])
        
        if col == ncols -1: # show colorbar only on the rightmost subplots
            # cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
            # Set tick labels in scientific notation with desired precision

            cbar = fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

            # Set font size for tick labels
            cbar.ax.tick_params(labelsize=8)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))  # Set limits for scientific notation
            cbar.ax.yaxis.set_major_formatter(formatter)

            cbar.ax.set_ylabel("Attention Scores", rotation=-90, va="bottom", fontsize=12)

        ax.set_title(title)

    # set overall title
    fig.suptitle(fig_title, fontsize=16)

    fig.tight_layout()
    plt.savefig(save_to + f'{fig_name}.png', dpi=1200, bbox_inches='tight')
    plt.show()



def get_split(example, chat:bool, prompt_position, tokenizer, attention_level, have_docid):
    question, documents = separate_question_and_context(example)

    prompt = get_qa_prompt(question, documents, prompt_position, attention_level, have_docid, replace_docid=None)
    if chat:
        prompt = format_chat_prompt(prompt)        
    
    print(prompt)

    tokens = tokenizer.tokenize(prompt, add_special_tokens=False)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    split_sents, split_ranges = sent_split(token_ids, tokenizer)
    return split_sents, split_ranges, token_ids, prompt


def main():
    # parse arguments
    args = parse_argument()


    save_to = "results/zero_shot/plots/weight_heatmap/"
    os.makedirs(save_to, exist_ok=True)

    fig_title = args.model
    fig_name = f"{args.model}_gold_vs_prompt_level_{args.attention_level}_docid_{args.have_docid}"

    adapter_path=None

    if "Llama" in args.model:
        model_name = "meta-llama/" + args.model
    elif "Mistral" in args.model:
        model_name = "mistralai/" + args.model
    elif "tulu" in args.model:
        model_name = "allenai/{}".format(args.model)
    

    model, tokenizer = load_model(model_name, adapter_path)

    # Modify your plotting code as follows:
    split_attn_list = []
    split_sents_list = []
    titles = []


    chat_flag = ("chat" in model_name)
    # chat_flag = False

    if chat_flag:
        selected_parts = [3, 4, 5, 6, 7] # TODO: check on the effect of selected parts on attention values
        split_sents = ["Instruction", "Doc 1", "Doc 2", "Doc 3", "Question"]
    else:
        selected_parts = [0, 1, 2, 3, 4]
        split_sents = ["Instruction", "Doc 1", "Doc 2", "Doc 3", "Question"]

    interval = args.num_docs // 3
    start_idx = interval//2
    gold_indexs = [start_idx, start_idx + interval, start_idx + 2*interval]

    gold_ticks = [f"Doc {gold_id+1}" for gold_id in gold_indexs]
    if args.attention_level == "token":
        parts = [str(gold_id+1) for gold_id in gold_indexs]
    
    elif args.attention_level == "position":
        parts = ["beginning", "midsection", "tail"]
        # parts = ["first", "second", "third"]
    
    print(parts)


    np.random.seed(0)
    sample_idxs = np.random.choice(2655, 10, replace=False)

    # sample_idxs = [17]

    for gold_doc_position in gold_indexs:
        input_path = f"data/nq_data/total_{args.num_docs}_documents_reverse/nq-open-{args.num_docs}_total_documents_gold_at_{gold_doc_position+1}_with_least_distractors.jsonl.gz"
        # load data and get the prompt
        inputs_docs = load_inputs(input_path)

        # example = inputs_docs[args.example_idx]
        split_attn_i = []
        split_sents_i = []
        titles_i = []
        
        for which_part in parts:
            all_posatt_diff_na = []
            for example_idx in sample_idxs:
                example = inputs_docs[example_idx]
                posatt_split_sents, posatt_split_ranges, posatt_token_ids, posatt_prompt = get_split(example, chat_flag, which_part, tokenizer, args.attention_level, args.have_docid)
                na_split_sents, na_split_ranges, na_token_ids, na_prompt = get_split(example, chat_flag, "na", tokenizer, args.attention_level, args.have_docid)

                posatt_tuned_attention = get_attention(posatt_prompt, model, tokenizer)
                na_tuned_attention = get_attention(na_prompt, model, tokenizer)

                torch.cuda.empty_cache()

                posatt_tuned_attention, posatt_tuned_split_attention = attention4d_to_2d(posatt_tuned_attention, posatt_split_ranges, selected_parts)
                na_tuned_attention, na_tuned_split_attention = attention4d_to_2d(na_tuned_attention, na_split_ranges, selected_parts)

                tuned_posatt_diff_na = np.array(posatt_tuned_split_attention) - np.array(na_tuned_split_attention)
                all_posatt_diff_na.append(tuned_posatt_diff_na)
            
            avg_posatt_diff_na = np.mean(all_posatt_diff_na, axis=0)
            # print(avg_posatt_diff_na.shape)
            split_attn_i.append(avg_posatt_diff_na)
            # split_sents_i.append(posatt_split_sents)
            split_sents_i.append(split_sents)
            # titles_i.append(f"gold_at_{gold_doc_position+1}_prompt_at_{which_part}_diff_na")
            titles_i.append("Gold Answer in " + gold_ticks[gold_indexs.index(gold_doc_position)] + " & Attention to " + gold_ticks[gold_indexs.index(gold_doc_position)])

        split_attn_list.extend(split_attn_i)
        split_sents_list.extend(split_sents_i)
        titles.extend(titles_i)

    plot_weights_subplots(split_attn_list, split_sents_list, titles, save_to, fig_name, fig_title, ncols=3)


if __name__ == "__main__":
    main()
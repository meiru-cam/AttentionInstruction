#!/usr/bin/env python3
import pathlib
from copy import deepcopy
from typing import List, Optional, Tuple, Type, TypeVar
import random

from pydantic.dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()

T = TypeVar("T")


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def get_qa_prompt(
    question: str, documents: List[Document], attention_to: str, attention_level: str, \
        have_docid: bool, replace_docid: bool, reverse_idx=False):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    
    prompt_filename = "qa_att_default.prompt"
    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    num_docs = len(documents)

    if attention_to == "random":
        att_prompt = "Focus on random parts of the search results when answering."
    elif attention_to == "all":
        att_prompt = "The answer is in the search results. Use the information from the search results as the main reference."
    elif attention_to != "na":
        if attention_level == "token":
            if not replace_docid:
                part_str = f"document {attention_to}"
            else:
                part_str = f"{attention_to} part"
        elif attention_level == "position":
            part_str = f"{attention_to} part"
        else:
            raise ValueError(f"Invalid attention level: {attention_level}")
        
        # this is the best prompt we found, idx_9 -> the 10th prompt in ../src/utils/attention_prompts.txt
        att_prompt = f" The answer is in the {part_str} of the search results. Use the information from the {part_str} of the search results as the main reference."
    else:
        att_prompt = ""

    prompt_template = prompt_template.split("\n\n{search_results}")[0] + att_prompt + "\n\n{search_results}" + prompt_template.split("\n\n{search_results}")[1]

    # Format the documents into strings
    formatted_documents = []

    index_to_word = {0: "beginning", 1: "midsection", 2: "tail"}

    for document_index, document in enumerate(documents):
        if attention_level == "token":
            if have_docid:
                if not replace_docid:
                    if reverse_idx:
                        formatted_documents.append(f"Document [{num_docs-document_index}](Title: {document.title}) {document.text}")
                    else:
                        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
                else:
                    if not reverse_idx:
                        if len(documents) == 9:
                            formatted_documents.append(f"{index_to_word[document_index//3]} (Title: {document.title}) {document.text}")
                        elif len(documents) == 3:
                            formatted_documents.append(f"{index_to_word[document_index]} (Title: {document.title}) {document.text}")
                    else:
                        if len(documents) == 9:
                            formatted_documents.append(f"{index_to_word[(num_docs-document_index-1)//3]} (Title: {document.title}) {document.text}")
                        elif len(documents) == 3:
                            formatted_documents.append(f"{index_to_word[num_docs-document_index-1]} (Title: {document.title}) {document.text}")
            else:
                formatted_documents.append(f"(Title: {document.title}) {document.text}")
        elif attention_level == "position":
            if have_docid:
                formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
            else:
                formatted_documents.append(f"(Title: {document.title}) {document.text}")
            
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))
    # return prompt_template.format(question=question, search_results=" ".join(formatted_documents))    


def get_closedbook_qa_prompt(question: str):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    with open(PROMPTS_ROOT / "closedbook_qa.prompt") as f:
        prompt_template = f.read().rstrip("\n")

    return prompt_template.format(question=question)


def get_kv_retrieval_prompt(
    data: List[Tuple[str, str]],
    key: str,
    query_aware_contextualization: bool = False,
):
    if not data:
        raise ValueError(f"Provided `data` must be truthy, got: {data}")
    if not key:
        raise ValueError(f"Provided `key` must be truthy, got: {key}")
    if key not in [x[0] for x in data]:
        raise ValueError(f"Did not find provided `key` {key} in data {data}")
    if len(data) != len(set([x[0] for x in data])):
        raise ValueError(f"`data` has duplicate keys: {data}")
    if len(data) < 2:
        raise ValueError(f"Must have at least 2 items in data: {data}")

    if query_aware_contextualization:
        with open(PROMPTS_ROOT / "kv_retrieval_with_query_aware_contextualization.prompt") as f:
            prompt_template = f.read().rstrip("\n")
    else:
        with open(PROMPTS_ROOT / "kv_retrieval.prompt") as f:
            prompt_template = f.read().rstrip("\n")

    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        start_character = "{" if index == 0 else " "
        data_string = f'"{record[0]}": "{record[1]}"'
        end_character = ",\n" if index != len(data) - 1 else "}"
        formatted_kv_records += start_character + data_string + end_character

    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)

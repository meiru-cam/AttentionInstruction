from copy import deepcopy
from typing import List, Optional, Tuple, Type, TypeVar

from pydantic.dataclasses import dataclass

import logging
from xopen import xopen
from tqdm import tqdm
import json
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

import argparse


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
    

def parse_args():
    parser = argparse.ArgumentParser(description="Generate NQ data with a specific gold index and distractors with descending relevance score.")
    parser.add_argument("--num_total_documents", type=int, default=3)
    parser.add_argument("--input_path", type=str, default="data/nq-open-30_total_documents_gold_at_0.jsonl.gz")
    parser.add_argument("--output_folder", type=str, default="data/nq_data/")
    return parser.parse_args()

def main():
    args = parse_args()

    output_folder = args.output_folder + f"total_{args.num_total_documents}_documents_reverse/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    intervals = args.num_total_documents//3
    start_idx = intervals//2
    gold_indexs = [start_idx, start_idx + intervals, start_idx + 2*intervals]

    for gold_index in gold_indexs:
        num_output_examples = 0
        output_path = output_folder + f"nq-open-{args.num_total_documents}_total_documents_gold_at_{gold_index+1}_with_least_distractors.jsonl.gz"
        with xopen(args.input_path) as fin, xopen(output_path, "w") as fout:
            for line in tqdm(fin):
                qa_retrieval_result = json.loads(line)
                # Get documents that don't contain the answer
                valid_distractors_with_retrieval_indices = [
                    (doc["original_retrieval_index"], doc) for idx, doc in enumerate(qa_retrieval_result["ctxs"]) if doc["hasanswer"] is False
                ]
                valid_distractors_with_retrieval_indices = valid_distractors_with_retrieval_indices[::-1]
                # Take the top `num_total_documents - 1` distractors
                distractor_docs_with_retrieval_indices = deepcopy(
                    valid_distractors_with_retrieval_indices[:args.num_total_documents - 1]
                )
                for original_retrieval_index, distractor_doc in distractor_docs_with_retrieval_indices:
                    distractor_doc["original_retrieval_index"] = original_retrieval_index
                    distractor_doc["isgold"] = False
                distractor_docs = [x[1] for x in distractor_docs_with_retrieval_indices]

                content_selection_example = deepcopy(qa_retrieval_result)
                gold_chunk = {
                    "title": qa_retrieval_result["nq_annotated_gold"]["title"],
                    "text": qa_retrieval_result["nq_annotated_gold"]["chunked_long_answer"],
                    "hasanswer": True,
                    "isgold": True,
                }
                ctxs = distractor_docs
                # Insert the gold chunk at thet specific index
                ctxs.insert(gold_index, gold_chunk)

                content_selection_example["ctxs"] = ctxs
                fout.write(json.dumps(content_selection_example) + "\n")
                num_output_examples += 1
        logger.info(f"Wrote {num_output_examples} output examples")


if __name__ == "__main__":
    main()
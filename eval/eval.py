"""
cd /env/lib/repos/wbr/repos_cloned/LMUnit-Develop/eval
conda activate wbr-dev
python eval.py --task flask --model_path /data/wberriosr/projects/lmunit/checkpoints/al_all_llama3.1_8b_sft_mse_pref.json/lm_1000 --tensor-parallel-size 2
python eval.py --task flask --model_path /data/shikib/checkpoints/al_all.json/lm_1000/ --tensor-parallel-size 4

"""

import json
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict

import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoTokenizer
from vllm import SamplingParams

from lmunit import LMUnit
from lmunit.constants import (DEFAULT_SYSTEM_PROMPT, DEFAULT_UNIT_TEST,
                              UNIT_TEST_PROMPT)
from lmunit.tasks import get_task, list_tasks


def unroll_preference_dataset(
    hf_dataset: Dataset, column_1: str, column_2: str
) -> Dataset:
    """Unroll a preference dataset by creating two rows for each original row,
    one for each preference column.

    Args:
        hf_dataset: The input HuggingFace Dataset
        column_1: First preference column name
        column_2: Second preference column name

    Returns:
        Dataset with unrolled preference data

    """
    df = hf_dataset.to_pandas()

    # Create two copies of the dataframe with vectorized operations
    df_1 = df.copy()
    df_1["response"] = df[column_1]
    df_1["index"] = df.index
    df_1["column"] = column_1

    df_2 = df.copy()
    df_2["response"] = df[column_2]
    df_2["index"] = df.index
    df_2["column"] = column_2

    # Concatenate the two dataframes efficiently
    new_df = pd.concat([df_1, df_2], ignore_index=True)
    return Dataset.from_pandas(new_df)


def roll_preference_dataset(hf_dataset: Dataset, column_1: str, column_2: str):
    df = hf_dataset.to_pandas()

    def agg_group(group: pd.DataFrame):
        base = group.iloc[0].drop(["response", "column", "score", "index"])
        scores = group.set_index("column")["score"]
        base["score_1"] = scores[column_1]
        base["score_2"] = scores[column_2]
        return base

    result_df = df.groupby("index").apply(agg_group).reset_index(drop=True)
    return Dataset.from_pandas(result_df)


def process_sample(
    sample: Dict, tokenizer: AutoTokenizer, use_rubric: bool, use_reference: bool
):
    query = sample[args.query_key]
    response = sample[args.response_key]
    unit_test = sample[args.unit_test_key]
    if use_rubric:
        rubric = sample["rubric"]
        unit_test = f"{unit_test}\n\nRubric: {rubric}"
    if use_reference:
        reference = sample["reference_answer"]
        unit_test = f"{unit_test}\n\nReference Answer: {reference}"
    message = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": UNIT_TEST_PROMPT.format(
                query=query, response=response, natural_unit_test=unit_test
            ),
        },
    ]
    return {
        "unit_test_prompt": tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
    }


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=list_tasks())
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--query-key", type=str, default="query")
    parser.add_argument("--response-key", type=str, default="response")
    parser.add_argument("--unit-test-key", type=str, default="natural_unit_test")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save outputs"
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    task = get_task(args.task)
    mode_eval = task.mode
    assert mode_eval in ["direct", "preference"], "Mode not supported"
    hf_dataset = load_dataset(task.dataset_name, split=task.split)
    if args.debug:
        hf_dataset = hf_dataset.select(range(10))
    model = LMUnit(args.model_path, args.tensor_parallel_size)
    metric_fn = task.metric_fn
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, logprobs=20)
    if mode_eval == "direct":
        dataset = hf_dataset.map(
            partial(
                process_sample,
                tokenizer=model.tokenizer,
                use_rubric=task.use_rubric,
                use_reference=task.use_reference,
            ),
            num_proc=os.cpu_count(),
        )
        outputs = model.generate(
            prompt=dataset["unit_test_prompt"], sampling_params=sampling_params
        )
        scores = [output["score"] for output in outputs]
        dataset = dataset.add_column("score", scores)

    if mode_eval == "preference":
        hf_dataset = hf_dataset.add_column(
            "natural_unit_test", [DEFAULT_UNIT_TEST] * len(hf_dataset)
        )
        dataset = unroll_preference_dataset(hf_dataset, task.column_1, task.column_2)
        dataset = dataset.map(
            partial(
                process_sample,
                tokenizer=model.tokenizer,
                use_rubric=task.use_rubric,
                use_reference=task.use_reference,
            ),
            num_proc=os.cpu_count(),
        )
        outputs = model.generate(
            prompt=dataset["unit_test_prompt"], sampling_params=sampling_params
        )
        scores = [output["score"] for output in outputs]
        dataset = dataset.add_column("score", scores)
        dataset = roll_preference_dataset(dataset, task.column_1, task.column_2)

    metrics = metric_fn(dataset, threshold=task.threshold)
    logger.info(metrics)

    # Save outputs if output directory is specified
    if args.output_dir:
        output_path = Path(args.output_dir) / args.task

        # Create directories
        dataset_dir = output_path / "hf_dataset"
        metrics_dir = output_path / "metrics"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        dataset.save_to_disk(str(dataset_dir))
        logger.info(f"Dataset saved to {dataset_dir}")

        # Save metrics
        metrics_file = metrics_dir / f"{args.task}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

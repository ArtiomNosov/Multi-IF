# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from tqdm import tqdm

from api_client import get_api_bot
from metrics import MultiTurnInstructionFollowingPromptSolution
from utils import GenerationSetting

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Отключить логи внешних библиотек
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('giga.client').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

lock = Lock()


def max_retry_wrapper(api_bot, messages, max_retry=3):
    for attempt in range(max_retry, 0, -1):
        try:
            response = api_bot.generate(messages)
            return response
        except Exception as e:
            print(messages)
            logger.error(f"API call failed with error: {e}. Retries left: {attempt - 1}")
            time.sleep(1)  # Brief pause before retrying
    return f'[MAX_RETRY=0] Failed.'


def process_row(api_bot, row, step, max_retry):
    try:
        if step == 1:
            messages = [json.loads(row['turn_1_prompt'])]
        else:
            messages = json.loads(row['turns']) + [
                {'role': 'assistant', 'content': row['responses']},
                json.loads(row[f'turn_{step}_prompt']),
            ]
        response = max_retry_wrapper(api_bot, messages, max_retry)
        updated_turns = json.dumps(messages)
        status = 'success' if not response.startswith('[MAX_RETRY') else 'failed'
        return updated_turns, response, status
    except Exception as e:
        logger.exception(f"Error processing row: {e}")
        print(row)
        return row.get('turns', '[]'), f'Exception: {e}', 'exception'


def step_fn_api(
    api_bot,
    input_df,
    step,
    need_write2file=True,
    output_filepath=None,
    max_retry=3,
    max_workers=5,  # Limit the number of threads
):
    total_loc = len(input_df)
    output_df = input_df.copy()
    with lock:
        if "turns" not in output_df.columns:
            output_df["turns"] = pd.array(["[]"] * len(output_df), dtype="string")
        if "responses" not in output_df.columns:
            output_df["responses"] = pd.array(["None"] * len(output_df), dtype="string")
        if "status" not in output_df.columns:
            output_df["status"] = pd.array(["pending"] * len(output_df), dtype="string")
        output_df['turn_index'] = step  # Update to current step

    rows_to_process = []
    for idx, row in input_df.iterrows():
        current_turn_index = row.get('turn_index', 0)
        response = row.get('responses', 'None')
        if current_turn_index > step or  (current_turn_index == step and not response.startswith('[MAX_RETRY')):
            print(f"Skipped idx: {idx}")
            continue  # Skip already processed rows
        rows_to_process.append((idx, row))
    logger.info(f"Processing {len(rows_to_process)} out of {total_loc} rows for step {step}")

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_row, api_bot, row, step, max_retry): idx
            for idx, row in rows_to_process
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx = future_to_idx[future]
            try:
                updated_turns, response, status = future.result()
                results[idx] = (updated_turns, response, status)
            except Exception as exc:
                logger.error(f"Row {idx} generated an exception: {exc}")
                results[idx] = (output_df.at[idx, "turns"], f'Exception: {exc}', 'exception')

    with lock:
        for idx, (turns, response, status) in results.items():
            output_df.at[idx, "turns"] = turns
            output_df.at[idx, "responses"] = response
            output_df.at[idx, "status"] = status

    if need_write2file and output_filepath:
        output_df.to_csv(output_filepath, index=False)
        logger.info(f"Step {step} results written to {output_filepath}")

    return output_df


def consolidate_results(api_model_name, output_filepath_prefix, steps):
    consolidated_df = None
    for step in steps:
        step_csv = f"data/results/{api_model_name}/{output_filepath_prefix}_step_{step}.csv"
        if os.path.exists(step_csv):
            temp_df = pd.read_csv(step_csv, keep_default_na=False)
            if consolidated_df is None:
                consolidated_df = temp_df.copy()
            else:
                # Merge on a unique identifier; assuming the index serves as a unique identifier
                consolidated_df = consolidated_df.combine_first(temp_df)
        else:
            logger.warning(f"Step {step} file {step_csv} does not exist and will be skipped.")

    if consolidated_df is not None:
        consolidated_csv = f"data/results/{api_model_name}/{output_filepath_prefix}_consolidated.csv"
        consolidated_df.to_csv(consolidated_csv, index=False)
        logger.info(f"All steps consolidated into {consolidated_csv}")
    else:
        logger.warning("No data available to consolidate.")


def main(
    api_model_name,
    input_data_csv: str = "dataset/multi_turn_sample.csv",
    generation_setting=GenerationSetting(
        max_new_tokens=25000, temperature=0.6, top_p=0.9, seed=42
    ),
    need_write2file: bool = True,
    output_filepath_prefix: str = "eval_result",
    max_workers: int = 5,  # Number of threads
    steps: list = [1, 2, 3],  # New parameter for steps
):
    benchmark_df = pd.read_csv(input_data_csv, keep_default_na=False)
    num_rows = len(benchmark_df.axes[0])
    logger.info(f"Number of rows in input data: {num_rows}")
    final_metric_result = {}

    api_bot = get_api_bot(api_model_name, generation_setting)
    step_input_df = benchmark_df.copy()
    for step in steps:  # Use the user-provided steps
        output_filepath = (
            f"data/results/{api_model_name}/{output_filepath_prefix}_step_{step}.csv"
        )
        os.makedirs(f'data/results/{api_model_name}', exist_ok=True)
        step_output_df = step_fn_api(
            api_bot=api_bot,
            input_df=step_input_df,
            step=step,
            need_write2file=need_write2file,
            output_filepath=output_filepath,
            max_workers=max_workers,
        )

        step_input_df = step_output_df.copy()
        step_metric_result = run_metric(
            api_model_name,
            output_filepath_prefix=output_filepath_prefix,
            step=step,
        )
        final_metric_result[step] = step_metric_result

    consolidate_results(api_model_name, output_filepath_prefix, steps=steps)


def run_metric(
    api_model_name,
    output_filepath_prefix: str = "eval_result",
    step: int = 1
):
    step_output_df = None
    step_csv = f"data/results/{api_model_name}/{output_filepath_prefix}_step_{step}.csv"
    if not os.path.exists(step_csv):
        logger.warning(f"CSV file {step_csv} does not exist and will be skipped.")
        return {}

    logger.info(f"Calculating metrics for step_{step}")
    step_output_df = pd.read_csv(step_csv, keep_default_na=False)

    if step_output_df is not None and not step_output_df.empty:
        metric_result = MultiTurnInstructionFollowingPromptSolution.metrics_gen(
            step_output_df
        )
        metric_result_df = pd.DataFrame.from_dict(metric_result, orient="index")
        metric_result_df.to_csv(f"data/results/{api_model_name}/{output_filepath_prefix}_step_{step}_metric.csv")
        logger.info(f"Step {step} metrics:\n{metric_result}")
        return metric_result
    else:
        logger.warning(f"No data available for step {step} to compute metrics.")
        return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model_name", type=str, default="o1-mini")
    parser.add_argument(
        "--input_data_csv", type=str, default="dataset/multi_turn_sample.csv"
    )
    parser.add_argument("--need_write2file", type=bool, default=True)
    parser.add_argument("--output_filepath_prefix", type=str, default="eval_result")

    parser.add_argument('--max_new_tokens', type=int, default=1024, help='o1 recommends 25000')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_workers', type=int, default=5, help='Number of threads for concurrency')

    # New steps argument
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='List of steps to process (e.g., --steps 1 2 3)'
    )

    args = parser.parse_args()
    logger.info(f'Args: \n  max_new_tokens: {args.max_new_tokens}, \n  temperature: {args.temperature}, \n  top_p: {args.top_p}, \n  seed: {args.seed}, \n  max_workers: {args.max_workers}, \n  steps: {args.steps}')

    if 'o1' in args.api_model_name:
        # o1 doesn't allow for customized top_p and temperature.
        args.top_p = 1
        args.temperature = 1
    generation_setting = GenerationSetting(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )

    main(
        api_model_name=args.api_model_name,
        input_data_csv=args.input_data_csv,
        generation_setting=generation_setting,
        need_write2file=args.need_write2file,
        output_filepath_prefix=args.output_filepath_prefix,
        max_workers=args.max_workers,
        steps=args.steps,
    )
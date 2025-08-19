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


from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import torch
import pandas as pd
import logging
from typing import List, Optional
from metrics import MultiTurnInstructionFollowingPromptSolution
import json 
import numpy as np
from dataclasses import dataclass
from vllm import LLM
from vllm.sampling_params import SamplingParams

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass 
class GenerationSetting:
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42

def get_inference_batch(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, input_df: pd.DataFrame, batch_size: int = 24, generation_setting: GenerationSetting = GenerationSetting(), need_write2file: bool = True, output_filepath: str = "generation_output.csv", device: Optional[str] = None)-> pd.DataFrame:
    """
        generate inference result given pre-trained model, tokenizer, and input dataframe
        the result will be written to the output_filepath if defined. by default, it will return 
        the generation output dataframe as the output
        Args:
            model: pre-trained model
            tokenizer: pre-trained tokenizer
            input_df: input dataframe
            batch_size: batch size for inference
            generation_setting: generation setting
            need_write2file: whether to write the result to the output_filepath
            output_filepath: output filepath
            device: device to run the inference on
    """
    output_df = input_df.copy()
    if "turns" not in output_df.columns:
        output_df.insert(1, "turns", value=pd.array(["None"] * len(output_df), dtype="string"))
    if "responses" not in output_df.columns:
        output_df.insert(1, "responses", value=pd.array(["None"] * len(output_df), dtype="string"))

    num_split = len(output_df)//batch_size + 1
    print(f"num_split = {num_split}")
    # 2: Apply the chat template
    for batch in np.array_split(output_df, num_split):
        logger.info(f"processing {len(batch)} input row. ")
        chat = batch["multi_turn_prompt_column"].tolist() 
        # 2: Apply the chat template
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
        tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)

        # 4: Generate text from the model
        # TODO: make generation setting configurable
        prefix = inputs['input_ids'].size(1)
        if tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else:
            terminators = [tokenizer.eos_token_id]
        outputs = model.generate(**inputs, max_new_tokens=generation_setting.max_new_tokens, eos_token_id=terminators, temperature=generation_setting.temperature, top_p=generation_setting.top_p)

        # 5: Decode the output back to a string
        decoded_output = tokenizer.batch_decode([output[prefix:] for output in outputs], skip_special_tokens=True)

        if len(batch) != len(decoded_output):
            raise ValueError(f"batch size {len(batch)} != decoded_output size {len(decoded_output)}")
        for index, row in batch.iterrows():
            output_df.loc[index, "turns"] = json.dumps(row["multi_turn_prompt_column"])
            output_df.loc[index, "responses"] = decoded_output.pop(0)


    if need_write2file:
        output_df.drop(columns=["multi_turn_prompt_column"], inplace=True)
        output_df.to_csv(output_filepath)

    return output_df

def get_inference_batch_vllm(
    model: LLM,  # Changed to use vllm's LLM
    tokenizer: PreTrainedTokenizerBase,
    input_df: pd.DataFrame,
    batch_size: int = 24,
    generation_setting: GenerationSetting = GenerationSetting(),
    need_write2file: bool = True,
    output_filepath: str = "generation_output.csv",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate inference results using vllm for faster generation.
    """
    output_df = input_df.copy()
    if "turns" not in output_df.columns:
        output_df.insert(1, "turns", value=pd.array(["None"] * len(output_df), dtype="string"))
    if "responses" not in output_df.columns:
        output_df.insert(1, "responses", value=pd.array(["None"] * len(output_df), dtype="string"))

    num_split = len(output_df) // batch_size + 1
    logger.info(f"Number of splits: {num_split}")

    # Prepare sampling parameters for vllm
    if tokenizer.convert_tokens_to_ids("<|eot_id|>"):
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        terminators = [tokenizer.eos_token_id]

    gen_params = SamplingParams(
        temperature=generation_setting.temperature,
        top_p=generation_setting.top_p,
        max_tokens=generation_setting.max_new_tokens,
        stop_token_ids=terminators,
    )

    # Use vllm for generation
    print_once = True
    for batch in np.array_split(output_df, num_split):
        logger.info(f"Processing {len(batch)} input rows.")
        chat = batch["multi_turn_prompt_column"].tolist()

        try:
            formatted_chat = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except:
            import ipdb; ipdb.set_trace()
        if print_once:
            print("Chat:", formatted_chat[0])
            print(gen_params)
            print_once=False
        # Generate outputs using vllm
        generation_outputs = model.generate(formatted_chat, sampling_params=gen_params)

        if len(batch) != len(generation_outputs):
            raise ValueError(
                f"Batch size {len(batch)} != number of generation outputs {len(generation_outputs)}"
            )

        for index, gen_output in zip(batch.index, generation_outputs):
            decoded_output = gen_output.outputs[0].text
            output_df.loc[index, "turns"] = json.dumps(batch.loc[index, "multi_turn_prompt_column"])
            output_df.loc[index, "responses"] = decoded_output.strip()

    if need_write2file:
        output_df.drop(columns=["multi_turn_prompt_column"], inplace=True)
        output_df.to_csv(output_filepath, index=False)

    return output_df

def preprocess_data(input_df: pd.DataFrame, prompt_columns: List[str], row_limit: int = -1)-> pd.DataFrame:
    """
    Preprocess the data by applying prompt reformatting
    input_df: the input dataframe
    row_limit: number of rows to process, -1 means all rows
    """
    new_prompt_column = MultiTurnInstructionFollowingPromptSolution.get_text_column_name()
    if new_prompt_column not in input_df.columns:
        input_df.insert(1, new_prompt_column, value=pd.array([None] * len(input_df), dtype="string"))
    else:
        raise ValueError(f"Column {new_prompt_column} already exists in the input dataframe!")

    # step 1: expanding/updating existing df 
    input_df = MultiTurnInstructionFollowingPromptSolution.expand_df(input_df)
    if row_limit > 0:
        input_df = input_df.head(row_limit)
    # step 2: apply prompt reformatting
    def reformat_prompt(df_row: pd.core.series.Series) -> str:
        return MultiTurnInstructionFollowingPromptSolution.reformat_prompt(
            df_row, prompt_columns
        ) 
    input_df[new_prompt_column] = input_df.apply(reformat_prompt, axis=1)
    return input_df




    

    


    

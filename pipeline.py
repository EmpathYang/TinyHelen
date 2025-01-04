import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import transformers
import torch
import json
import warnings
import re
from copy import deepcopy

from utils import load_jsonl, GENERATION_DIR, CURRENT_DIR

def get_model_generation(model_id, prompts):
    def prompt_generator(prompts):
        for p in prompts:
            yield p

    prompts = [f"### TASK INPUT:\n{p}\n\n### RESPONSE:\n" for p in prompts]
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16, "max_length": 1024, "pad_token_id": 0, "eos_token_id": 0}, device_map="auto")

    responses = [r[0]["generated_text"] for r in pipeline(prompt_generator(prompts), batch_size=128)]
    return responses

def extract_digits(input_string):
    digits = re.findall(r'\d+', input_string)[0]
    return digits

def find_closest_checkpoint_dir(directory, number):
    closest_dir = None
    min_difference = float('inf')
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if d.startswith('checkpoint-'):
                try:
                    suffix = int(d.split('-')[-1])
                    difference = abs(suffix - number)
                    if difference < min_difference:
                        min_difference = difference
                        closest_dir = os.path.join(root, d)
                except ValueError:
                    continue

    return closest_dir

model_ids = []
os.makedirs(GENERATION_DIR, exist_ok=True)
question_objs = load_jsonl("data/eval/all.jsonl")
questions = [question_obj["question"] for question_obj in question_objs]
for model_id in model_ids:
    responses = get_model_generation(model_id=model_id, prompts=questions)
    basename = os.path.basename(model_id) if "checkpoint" not in os.path.basename(model_id) else os.path.basename(os.path.dirname(model_id)) + f"-{extract_digits(os.path.basename(model_id))}"
    with open(os.path.join(GENERATION_DIR, f"{basename}.jsonl"), "a") as af:
        for question_obj, response in zip(question_objs, responses):
            json_obj = deepcopy(question_obj)
            assert json_obj["question"] in response
            json_obj["model"] = basename
            json_obj["response"] = response
            af.write(f"{json.dumps(json_obj)}\n")
import re
from nltk.tokenize import sent_tokenize
import os
import random
from tqdm import tqdm
import json
from gpt import get_gpt_response
from transformers import GPT2Tokenizer, set_seed
import zstandard as zstd
import hashlib
from copy import deepcopy
import logging
import time
import torch
from functools import partial
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data", "leaner")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "data")
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "prompts", "simplification")
INSTRUCTION_FOLLOWING_QUESTIONS_DIR = os.path.join(CURRENT_DIR, "data", "leaner", "eval")
GENERATION_DIR = os.path.join(CURRENT_DIR, "generation")
GENERATION_EVALUATION_DIR = os.path.join(CURRENT_DIR, "generation_evaluation")
CACHE_DIR = None
LOSS_RANGE = [0, 10]
NUM_BINS = 500
PLOT_NUM_BINS = 400


def fixed_hash(input_string):
    hasher = hashlib.sha256()
    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

set_seed(65)

CORPUS_SPLITS = ["train", "validation", "test"]
SPLIT_COMPOSITION = {
    "train": 1,
    "validation": 0.05,
    "test":0.05,
}
CORPUS_COMPOSITION = {
    "web": 0.6,
    "book": 0.05,
    "wiki": 0.025,
    "textbook": 0.075,
    "conversation": 0.025,
    "code": 0.175,
    "math": 0.025,
}

NATURAL_LANGUAGE_SET_SPLITS = ["web", "book", "wiki", "textbook", "conversation"]
FORMAL_LANGUAGE_SET_SPLITS = ["code", "math"]
INSTRUCTION_SET_SPLITS = ["instruction"]

SET_SPLIT_JSON_KEY_MAP = {
    "web": ["text", "set_split"],
    "book": ["text", "set_split"],
    "wiki": ["text", "set_split"],
    "textbook": ["text", "set_split"],
    "conversation": ["text", "set_split"]
}

SET_SPLIT_JSON_TEXT_FIELD_MAP = {
    "web": ["text"],
    "book": ["text"],
    "wiki": ["text"],
    "textbook": ["text"],
    "conversation": ["text"],
    "code": ["prompt", "response"],
    "math": ["problem", "hints"],
    "instruction": ["instruction", "input", "output"]
}

SET_SPLIT_COLOR_MAP = {
    "web": "#267365",
    "book": "#C37EDB",
    "wiki": "#9AEBA3",
    "textbook": "#8F797E",
    "conversation": "#FF6B1A",
    "code": "#D94169",
    "math": "#2A4359",
    "instruction": "#A62145",
}

SET_SPLIT_PROMPT_TEMPLATE_MAP = {
    "code": "".join(open(os.path.join(CURRENT_DIR, "prompts", "templates", "code_template.txt"), "r")),
    "math": "".join(open(os.path.join(CURRENT_DIR, "prompts", "templates", "math_template.txt"), "r")),
    "instruction": "".join(open(os.path.join(CURRENT_DIR, "prompts", "templates", "instruction_template.txt"), "r"))
}

TOTAL_TOKEN_NUM_FILEDIR_MAP = {
        "10M": ["10M"],
        "100M": ["10M", "100M"],
        "1B": ["10M", "100M", "1B"],
        "10B": ["10M", "100M", "1B", "10B"],
    }

def parse_elements(text, key_list):
        element_dict = {}
        for k in key_list:
            _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\s]*:|$)', text, re.DOTALL)
            element_dict[k] = _match.group(1).strip() if _match else ""
        return element_dict

def unify_set_split_text_field(examples, set_split="N/A", require_hash=False, require_text_for_hash=False):
    if "text" in examples.keys():
        new_examples = [e for e in examples["text"]]
        text_for_hash = [e for e in examples["text"]]
    elif set_split == "code":
        new_examples = []
        text_for_hash = []
        for p, r in zip(examples["prompt"], examples["response"]):
            prompt = SET_SPLIT_PROMPT_TEMPLATE_MAP["code"].replace("{input}", p).replace("{response}", r)
            new_examples.append(prompt)
            text_for_hash.append(p + " " + r)
    elif set_split == "math":
        new_examples = []
        text_for_hash = []
        for p, hs in zip(examples["problem"], examples["hints"]):
            h = "\n".join(hs)
            prompt = SET_SPLIT_PROMPT_TEMPLATE_MAP["math"].replace("{input}", p).replace("{response}", h)
            new_examples.append(prompt)
            text_for_hash.append(p + " " + " ".join(hs))
    elif set_split == "instruction":
        new_examples = []
        text_for_hash = []
        for ins, inp, outp in zip(examples["instruction"], examples["input"], examples["output"]):
            p = ins + "\n" + inp if inp else ins
            prompt = SET_SPLIT_PROMPT_TEMPLATE_MAP["instruction"].replace("{input}", p).replace("{response}", outp)
            new_examples.append(prompt)
            text_for_hash.append(p + " " + outp)

    if "set_split" in examples.keys():
        set_splits = [e for e in examples["set_split"]]
    else:
        set_splits = [set_split]*len(new_examples)

    if require_hash:
        return {"text": new_examples, "set_split": set_splits, "hash": [e for e in examples["hash"]]}
    if require_text_for_hash:
        return {"text": new_examples, "set_split": set_splits, "text_for_hash": text_for_hash}
    return {"text": new_examples, "set_split": set_splits}

def estimate_token_num(examples):
    return {**examples, "token_num": [len(e.split()) for e in examples["text"]]}

def estimate_hist(examples):
    return {**examples, "hist": [np.histogram([loss] * token_num, bins=NUM_BINS, range=LOSS_RANGE)[0].tolist() for loss, token_num in zip(examples["loss"], examples["token_num"])]}

def get_json_concatenated_textual_fields(json_obj, set_split):
    textual_field_list = []
    for field in SET_SPLIT_JSON_TEXT_FIELD_MAP[set_split]:
        text = json_obj[field]
        if isinstance(text, str) and text != "":
            textual_field_list.append(text)
        if isinstance(text, list):
            textual_field_list += text
    if len(textual_field_list) > 0:
        return " ".join(textual_field_list)
    return None

def get_set_split_statistics(general_data_dir, corpus_total_token_num_choices=["10M"], keep_previous=False):
    import csv

    class Table:
        def __init__(self):
            self.data = {}

        def set_cell(self, row, col, value):
            if row not in self.data:
                self.data[row] = {}
            self.data[row][col] = value

        def get_cell(self, row, col):
            return self.data.get(row, {}).get(col, None)

        def export_csv(self, filename):
            with open(filename, 'w') as csvfile:
                writer = csv.writer(csvfile)
                # Write header row with column names
                cols = sorted(set(col for row in self.data.values() for col in row))
                writer.writerow([''] + cols)

                # Write data rows
                for row_key in sorted(self.data.keys()):
                    row_data = [self.get_cell(row_key, col) for col in cols]
                    writer.writerow([row_key] + row_data)

        def import_csv(self, filename):
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)  # Read the header row
                cols = header[1:]  # Exclude the first column (row keys)
                
                for row in reader:
                    row_key = row[0]
                    for i, col in enumerate(cols):
                        if row_key not in self.data:
                            self.data[row_key] = {}
                        self.data[row_key][col] = row[i + 1]

        def print_table(self):
            # Print table to console (for demonstration)
            cols = sorted(set(col for row in self.data.values() for col in row))
            header = [''] + cols
            print('\t'.join(header))
            for row_key in sorted(self.data.keys()):
                row_data = [self.get_cell(row_key, col) for col in cols]
                print(f'{row_key}\t' + '\t'.join(map(str, row_data)))

    table = Table()
    if keep_previous and os.path.exists(os.path.join(general_data_dir, "statistics.csv")):
        table.import_csv(os.path.exists(os.path.join(general_data_dir, "statistics.csv")))

    for corpus_total_token_num_choice in corpus_total_token_num_choices:
        for corpus_split in CORPUS_SPLITS:
            data_dir = os.path.join(general_data_dir, corpus_total_token_num_choice, corpus_split)
            for set_split in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS:
                if not keep_previous and table.get_cell(set_split, f"{corpus_total_token_num_choice}/{corpus_split}"):
                    continue
                token_sum = 0
                filepaths = get_set_split_valid_filepaths(data_dir, set_split)
                for filepath in filepaths:
                    with open(filepath, 'r') as f_in:
                        for line in tqdm(f_in, desc=filepath):
                            try:
                                json_obj = json.loads(line)
                                concatenated_textual_fields = get_json_concatenated_textual_fields(json_obj, set_split)
                                if concatenated_textual_fields is not None:
                                    token_sum += count_tokens_in_text(tokenizer, concatenated_textual_fields)
                            except json.JSONDecodeError:
                                pass
                table.set_cell(set_split, f"{corpus_total_token_num_choice}/{corpus_split}", token_sum)
                table.export_csv(os.path.join(general_data_dir, "statistics.csv"))

def tokenize_sentences(paragraph, package="nltk"):
    if package == "nltk":
        return sent_tokenize(paragraph)
    elif package == "spacy":
        doc = nlp(paragraph)
        sentences = [sent.text for sent in doc.sents]
        return sentences

def is_valid_leaner_filename(filename):
    def substring_before_digits(s):
        match = re.search(r'^\D+', s)
        if match:
            return match.group(0)
        return ""
    if substring_before_digits(filename) in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS + INSTRUCTION_SET_SPLITS and filename.endswith(".jsonl") and filename.split(".")[0][len(substring_before_digits(filename)):].isdigit():
        return True
    return False

def is_valid_original_filename(filename):
    def substring_before_digits(s):
        match = re.search(r'^\D+', s)
        if match:
            return match.group(0)
        return ""
    if substring_before_digits(filename) in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS + INSTRUCTION_SET_SPLITS and (filename.endswith(".zst") or filename.endswith(".jsonl")) and filename.split(".")[0][len(substring_before_digits(filename)):].isdigit():
        return True
    return False

def count_tokens_in_text(tokenizer, text):
    return len(tokenizer.tokenize(text))

def count_tokens_in_file(tokenizer, filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    tokens = tokenizer.tokenize(text)
    return len(tokens)

def count_tokens_in_dir(input_dir):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    total_tokens = 0

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        if os.path.isfile(filepath) and filename.endswith('.txt'):
            total_tokens += count_tokens_in_file(tokenizer, filepath)

    return total_tokens

def count_tokens_in_leaner_file(filepath, set_split):
    total_token = 0
    with open(filepath, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if set_split == "code":
                text = " ".join([json_obj[tf] for tf in ["prompt", "response"] if json_obj[tf] != ""])
            elif set_split == "math":
                text = " ".join([json_obj["problem"], "\n".join(json_obj["hints"])])
            elif set_split == "instruction":
                text = " ".join([json_obj[tf] for tf in ["instruction", "input", "output"] if json_obj[tf] != ""])
            else:
                text = json_obj.get("text")

            if text.strip() is not None:
                total_token += count_tokens_in_text(tokenizer=tokenizer, text=text)
    return total_token

def count_tokens_in_leaner(directory, set_split):
    total_token = 0
    for filename in os.listdir(directory):
        if filename.startswith(set_split) and filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            total_token += count_tokens_in_leaner_file(filepath=filepath, set_split=set_split)
    return total_token

def compute_loss(article, model, tokenizer):
    from torch.nn import CrossEntropyLoss
    model_config = getattr(model, "config", None) if getattr(model, "config", None) else getattr(model.module, "config", None)
    model_device = getattr(model, "device", None) if getattr(model, "device", None) else getattr(model.module, "device", None)
    max_seq_length = model_config.max_position_embeddings
    batch_size = 4
    tokenized_article = tokenizer.encode(article + tokenizer.eos_token, add_special_tokens=False)
    total_tokens = len(tokenized_article)
    attention_masks = [1] * total_tokens
    labels = tokenized_article
    segments = [tokenized_article[i:i+max_seq_length] for i in range(0, total_tokens, max_seq_length)]
    attention_masks = [attention_masks[i:i+max_seq_length] for i in range(0, total_tokens, max_seq_length)]
    labels = [labels[i:i+max_seq_length] for i in range(0, total_tokens, max_seq_length)]
    batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]
    attention_masks = [attention_masks[i:i+batch_size] for i in range(0, len(segments), batch_size)]
    labels = [labels[i:i+batch_size] for i in range(0, len(segments), batch_size)]
    if len(batches[-1][-1]) < max_seq_length:
        pad_length = max_seq_length - len(batches[-1][-1])
        batches[-1][-1] += [tokenizer.pad_token_id] * pad_length
        attention_masks[-1][-1] += [0] * pad_length
        labels[-1][-1] += [-100] * pad_length

    total_loss = 0.0
    
    loss_list = []
    with torch.no_grad():
        for input_ids, attention_mask, label in zip(batches, attention_masks, labels):
            input_ids = torch.tensor(input_ids).to(model_device)
            attention_mask = torch.tensor(attention_mask).to(model_device)
            label = torch.tensor(label).to(model_device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, model_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            losses = loss_fct(shift_logits, shift_labels)[torch.where(shift_labels!=-100)]

            total_loss += losses.sum().item()
            loss_list += losses.cpu().numpy().tolist()

        avg_loss = total_loss / total_tokens

        loss_range = LOSS_RANGE
        num_bins = NUM_BINS
        hist, _ = np.histogram(loss_list, bins=num_bins, range=loss_range)
        hist = hist.tolist()

    return avg_loss, total_tokens, loss_list, hist

def get_text_from_leaner_obj(leaner_obj, set_split):
    if set_split in NATURAL_LANGUAGE_SET_SPLITS+FORMAL_LANGUAGE_SET_SPLITS+INSTRUCTION_SET_SPLITS:
        try:
            return leaner_obj["text"]
        except:
            print(set_split, leaner_obj)
    else:
        raise NotImplementedError()

def action_strip_text(json_obj, set_split, **kwargs):
    json_obj["text"] = json_obj["text"].strip()
    return json_obj

def action_pop_keys(json_obj, set_split, **kwargs):
    keys = ["ori prompt", "ori response"]
    for key in keys:
        try:
            json_obj.pop(key)
        except:
            continue
    return json_obj

def action_finish_code(json_obj, set_split, **kwargs):
    if json_obj["response"] == "b"*256:
        new_json_obj = deepcopy(json_obj)
        _prompt = json_obj["prompt"]
        try:
            assert "[RESPONSE]" in _prompt and _prompt.endswith("<end>")
            _edited = _prompt.split("[RESPONSE]")
            _task, _response = _edited[0].strip(), _edited[1][:-len("<end>")].strip()
            new_json_obj["prompt"], new_json_obj["response"] = _task, _response
            return new_json_obj
        except:
            try:
                res = get_gpt_response(f'''Finish the following; don't repeat my input, directly continue the text and end your response with "<end>": {_prompt}''')
                new_res = _prompt + res
                assert ("<split>" in new_res or "[RESPONSE]" in new_res) and new_res.endswith("<end>")
                if "<split>" in new_res:
                    _edited = new_res.split("<split>")
                elif "[RESPONSE]" in new_res:
                    _edited = new_res.split("[RESPONSE]")
                _task, _response = _edited[0].strip(), _edited[1][:-len("<end>")].strip()
                new_json_obj["prompt"], new_json_obj["response"] = _task, _response
                return new_json_obj
            except:
                return None
    else:
        return json_obj

def action_compute_loss(json_obj, set_split, **kwargs):
    if "save" in kwargs.keys() and kwargs["save"]:
        json_obj["loss"], json_obj["token_num"], _, json_obj["hist"] = compute_loss(article=json_obj["text"], model=kwargs["model"], tokenizer=kwargs["tokenizer"])
    else:
        json_obj["loss"], _, _, _ = compute_loss(article=json_obj["text"], model=kwargs["model"], tokenizer=kwargs["tokenizer"])
    return json_obj

def action_add_hash(examples):
    return {**examples, "hash": [fixed_hash(e) for e in examples["text"]]}

def action_store_loss(json_obj, **kwargs):
    store_text = kwargs.get("store_text", False)
    new_json_obj = deepcopy(json_obj)
    new_json_obj["hash"] = fixed_hash(new_json_obj["text"])
    if not store_text:
        del new_json_obj["text"]
    del_keys = set(json_obj.keys()) - set(["text", "set_split", "loss", "token_num", "hist", "hash"])
    for k in del_keys:
        del new_json_obj[k]
    with open(kwargs["output_filepath"], "a") as af:
        try:
            af.write(f"{json.dumps(new_json_obj)}\n")
        except:
            af.write(f"{json.dumps(dict(new_json_obj))}\n")

def action_delete_curriculum_training_irrelevant_keys(json_obj, set_split, **kwargs):
    del_keys = set(json_obj.keys()) - set(["text", "set_split", "loss", "token_num", "hist", "hash"])
    for k in del_keys:
        del json_obj[k]
    return json_obj

def action_process_instructions(json_obj, set_split, **kwargs):
    def remove_between(string, a, b):
        while True:
            start = string.find(a)
            if start == -1:
                break
            end = string.find(b, start)
            if end == -1:
                break
            string = string[:start] + string[end:]
        return string
    def remove_before_edited(string):
        index = string.find("[EDITED]")
        if index != -1:
            return string[index + len("[EDITED]"):].strip()
        else:
            return string.strip()
    _instruction = json_obj["instruction"].strip()
    _input = json_obj["input"]
    _output = json_obj["output"]
    if len(_output) > 0:
        if _instruction.startswith("[TASK]"):
            _instruction = _instruction.strip("[TASK]")
        if "[TASK]" in _instruction and "[INPUT]" in _instruction:
            _instruction = remove_between(_instruction, "[TASK]", "[INPUT]")
        if "[INPUT]" in _instruction and "[RESPONSE]" in _instruction:
            _instruction = remove_between(_instruction, "[INPUT]", "[RESPONSE]")
        if "[RESPONSE]" in _instruction and "[EDITED]" in _instruction:
            _instruction = remove_between(_instruction, "[RESPONSE]", "[EDITED]")
        elif "[RESPONSE]" in _instruction and not "[EDITED]" in _instruction:
            _instruction = _instruction[:_instruction.index("[RESPONSE]")]
        if "[EDITED]" in _instruction:
            _instruction = remove_before_edited(_instruction).strip("[TASK]")
        if "[RESPONSE]" in _output:
            _output_split = _output.split("[RESPONSE]")
            _input = _output_split[0].strip("[INPUT]").strip()
            _input = _input if (_input != "none" and _input != "None") else ""
            _output = _output_split[1].strip()
    if _instruction and _output:
        json_obj["instruction"] = _instruction
        json_obj["input"] = _input
        json_obj["output"] = _output
        return json_obj
    return None

def action_expression_replacement(json_obj, set_split, **kwargs):
    def replace_if_not_followed_by_alnum(input_str, old, new):
        pattern = re.compile(f'{re.escape(old)}(?![a-zA-Z0-9])')
        result = pattern.sub(new, input_str)
        return result
    replacement_dict = kwargs["replacement_dict"]
    for old_expression in replacement_dict.keys():
        for text_field in SET_SPLIT_JSON_TEXT_FIELD_MAP[set_split]:
            if isinstance(json_obj[text_field], str):
                json_obj[text_field] = replace_if_not_followed_by_alnum(json_obj[text_field], old_expression, replacement_dict[old_expression])
            elif isinstance(json_obj[text_field], list):
                json_obj[text_field] = [replace_if_not_followed_by_alnum(t, old_expression, replacement_dict[old_expression]) for t in json_obj[text_field]]
    return json_obj

countries_from_gpt = [
    "Afghanistan", "Albania", "Algeria", "America", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]


adjectives_for_countries_from_gpt = [
    "Afghan", "Albanian", "Algerian", "American", "Andorran", "Angolan", "Antiguan and Barbudan", "Argentinian", "Armenian", "Australian", "Austrian", "Azerbaijani", "Bahamian", "Bahraini", "Bangladeshi", "Barbadian", "Belarusian", "Belgian", "Belizean", "Beninese", "Bhutanese", "Bolivian", "Bosnian and Herzegovinian", "Botswanan", "Brazilian", "Bruneian", "Bulgarian", "Burkinabe", "Burundian", "Cape Verdean", "Cambodian", "Cameroonian", "Canadian", "Central African", "Chadian", "Chilean", "Chinese", "Colombian", "Comorian", "Congolese", "Costa Rican", "Ivorian", "Croatian", "Cuban", "Cypriot", "Czech", "Danish", "Djiboutian", "Dominican", "Ecuadorian", "Egyptian", "Salvadoran", "Equatorial Guinean", "Eritrean", "Estonian", "Swazi", "Ethiopian", "Fijian", "Finnish", "French", "Gabonese", "Gambian", "Georgian", "German", "Ghanaian", "Greek", "Grenadian", "Guatemalan", "Guinean", "Guinea-Bissauan", "Guyanese", "Haitian", "Honduran", "Hungarian", "Icelandic", "Indian", "Indonesian", "Iranian", "Iraqi", "Irish", "Israeli", "Italian", "Jamaican", "Japanese", "Jordanian", "Kazakhstani", "Kenyan", "I-Kiribati", "North Korean", "South Korean", "Kosovar", "Kuwaiti", "Kyrgyzstani", "Laotian", "Latvian", "Lebanese", "Mosotho", "Liberian", "Libyan", "Liechtensteiner", "Lithuanian", "Luxembourger", "Malagasy", "Malawian", "Malaysian", "Maldivian", "Malian", "Maltese", "Marshallese", "Mauritanian", "Mauritian", "Mexican", "Micronesian", "Moldovan", "Monégasque", "Mongolian", "Montenegrin", "Moroccan", "Mozambican", "Myanmar", "Namibian", "Nauruan", "Nepali", "Dutch", "New Zealand", "Nicaraguan", "Nigerien", "Nigerian", "North Macedonian", "Norwegian", "Omani", "Pakistani", "Palauan", "Palestinian", "Panamanian", "Papua New Guinean", "Paraguayan", "Peruvian", "Philippine", "Polish", "Portuguese", "Qatari", "Romanian", "Russian", "Rwandan", "Saint Kitts and Nevis", "Saint Lucian", "Saint Vincent and the Grenadines", "Samoan", "San Marinese", "Sao Tomean", "Saudi Arabian", "Senegalese", "Serbian", "Seychellois", "Sierra Leonean", "Singaporean", "Slovak", "Slovenian", "Solomon Islander", "Somali", "South African", "South Sudanese", "Spanish", "Sri Lankan", "Sudanese", "Surinamese", "Swedish", "Swiss", "Syrian", "Tajikistani", "Tanzanian", "Thai", "Timorese", "Togolese", "Tongan", "Trinidadian and Tobagonian", "Tunisian", "Turkish", "Turkmen", "Tuvaluan", "Ugandan", "Ukrainian", "Emirati", "British", "American", "Uruguayan", "Uzbekistani", "Ni-Vanuatu", "Vatican", "Venezuelan", "Vietnamese", "Yemeni", "Zambian", "Zimbabwean"
]

cities_from_gpt = ["New York", "New York City", "Los Angeles", "Chicago", "Las Vegas", "San Francisco", "Miami", "Washington", "Washington, D.C.", "Toronto", "Vancouver", "Montreal", "Mexico City", "Rio de Janeiro", "Buenos Aires", "Santiago", "Lima", "London", "Paris", "Berlin", "Madrid", "Rome", "Barcelona", "Amsterdam", "Moscow", "St. Petersburg", "Stockholm", "Oslo", "Copenhagen", "Athens", "Istanbul", "Dubai", "Cairo", "Tel Aviv", "Mumbai", "New Delhi", "Beijing", "Shanghai", "Tokyo", "Seoul", "Sydney", "Melbourne", "Auckland"]

action_expression_replacement_adapt = partial(action_expression_replacement, replacement_dict={
    **{
        k: k[:2].upper() + " country" for k in countries_from_gpt
    },
    **{
        k: k[:2].upper() + " country" for k in adjectives_for_countries_from_gpt
    },
    **{
        k: k[:2].upper() + " city" for k in countries_from_gpt
    }})

def process_file_in_leaner(filepath, set_split, action, **kwargs):
    basename_without_extension = os.path.basename(filepath).split(".")[0].split("_")[0]
    if "output_dir" in kwargs.keys():
        output_dir = kwargs["output_dir"]
    else:
        output_dir = os.path.dirname(filepath)
    if "output_filename_suffix" in kwargs.keys():
        output_filename_suffix = kwargs["output_filename_suffix"]
        if output_filename_suffix and output_dir != os.path.dirname(filepath):
            output_filepath = os.path.join(output_dir, basename_without_extension) + f"_{output_filename_suffix}.jsonl"
        else:
            output_filepath = os.path.join(output_dir, basename_without_extension) + ".jsonl"
    else:
        output_filepath = os.path.join(output_dir, basename_without_extension) + "_processed.jsonl"
    with open(filepath, 'r') as f_in:
        skip_by_text = False
        if os.path.exists(output_filepath) and "skip_by_text" in kwargs.keys() and kwargs["skip_by_text"] == True:
            skip_by_text = True
            skip_by_text_set = set([json_obj["text"] for json_obj in load_jsonl(output_filepath)])
        else:
            with open(output_filepath, 'w') as f_out:
                pass
        
        for line in f_in:
            try:
                json_obj = json.loads(line)
                if skip_by_text and json_obj["text"] in skip_by_text_set:
                    continue
                processed_json_obj = action(json_obj, set_split, **kwargs)
                if processed_json_obj is not None:
                    with open(output_filepath, 'a') as f_out:
                        f_out.write(f"{json.dumps(processed_json_obj)}\n")
            except json.JSONDecodeError:
                pass

def process_leaner(general_data_dir, action, corpus_total_token_num_choices=["10M"], **kwargs):
    for corpus_total_token_num_choice in corpus_total_token_num_choices:
        for corpus_split in CORPUS_SPLITS:
            for set_split in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS:
                filepaths = get_set_split_valid_filepaths(os.path.join(general_data_dir, corpus_total_token_num_choice, corpus_split), set_split)
                for filepath in filepaths:
                    process_file_in_leaner(filepath=filepath, set_split=set_split, action=action, **kwargs)

def deduplicate_file_in_leaner(filepath, **kwargs):
    from collections import OrderedDict
    basename_without_extension = os.path.basename(filepath).split(".")[0].split("_")[0]
    if "output_dir" in kwargs.keys():
        output_dir = kwargs["output_dir"]
    else:
        output_dir = os.path.dirname(filepath)
    if "output_filename_suffix" in kwargs.keys():
        output_filename_suffix = kwargs["output_filename_suffix"]
        output_filepath = os.path.join(output_dir, basename_without_extension) + f"_{output_filename_suffix}.jsonl"
    else:
        output_filepath = os.path.join(output_dir, basename_without_extension) + "_processed.jsonl"
    hash_dict = OrderedDict()
    with open(filepath, 'r') as f_in:
        for line in f_in:
            try:
                json_obj = json.loads(line)
                hash_dict[json_obj["hash"]] = json_obj
            except json.JSONDecodeError:
                pass
    with open(output_filepath, 'w') as f_out:
        pass
    for hash in hash_dict:
        with open(output_filepath, 'a') as f_out:
            f_out.write(f"{json.dumps(hash_dict[hash])}\n")

def deduplicate_files_in_leaner(filepaths, **kwargs):
    basename = os.path.basename(filepaths[0])
    set_split = [set_split for set_split in NATURAL_LANGUAGE_SET_SPLITS+FORMAL_LANGUAGE_SET_SPLITS if basename.startswith(set_split)][0]
    if "output_dir" in kwargs.keys():
        output_dir = kwargs["output_dir"]
    else:
        output_dir = os.path.dirname(filepaths[0])
    if "output_filename_suffix" in kwargs.keys():
        output_filename_suffix = kwargs["output_filename_suffix"]
    else:
        output_filename_suffix = "processed"
    from collections import OrderedDict
    hash_dict = OrderedDict()
    for filepath in filepaths:
        with open(filepath, 'r') as f_in:
            for line in f_in:
                try:
                    json_obj = json.loads(line)
                    hash_dict[json_obj["hash"]] = json_obj
                except json.JSONDecodeError:
                    pass
    token_num_buffer = 0
    set_split_idx = 0
    for hash in hash_dict:
        tokens_in_text = count_tokens_in_text(tokenizer, get_json_concatenated_textual_fields(hash_dict[hash], set_split))
        token_num_buffer += tokens_in_text
        if token_num_buffer > 10e6:
            token_num_buffer = 0
            set_split_idx += 1
        with open(os.path.join(output_dir, "{}{:04d}_{}.jsonl".format(set_split, set_split_idx, output_filename_suffix)), "a") as f_out:
            f_out.write(f"{json.dumps(hash_dict[hash])}\n")

def deduplicate_leaner(general_data_dir, corpus_total_token_num_choices=["10M"]):
    for corpus_total_token_num_choice in corpus_total_token_num_choices:
        for corpus_split in CORPUS_SPLITS:
            for set_split in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS:
                filepaths = get_set_split_valid_filepaths(os.path.join(general_data_dir, corpus_total_token_num_choice, corpus_split), set_split)
                deduplicate_files_in_leaner(filepaths)

def process_set_split_in_leaner(directory, set_split, action, **kwargs):
    for filename in os.listdir(directory):
        if filename.startswith(set_split) and filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            if "output_filename_suffix" in kwargs.keys():
                output_filename_suffix = kwargs["output_filename_suffix"]
                output_filepath = os.path.splitext(filepath)[0] + f"_{output_filename_suffix}.jsonl"
            else:
                output_filepath = os.path.splitext(filepath)[0] + "_processed.jsonl"
            with open(filepath, 'r') as f_in:
                for line in f_in:
                    try:
                        json_obj = json.loads(line)
                        processed_json_obj = action(json_obj, set_split, kwargs)
                        if processed_json_obj is not None:
                            with open(output_filepath, 'a') as f_out:
                                f_out.write(f"{json.dumps(processed_json_obj)}\n")
                    except json.JSONDecodeError:
                        pass

def make_leaner_copy(general_data_dir, backup_data_dir, corpus_total_token_num_choices=["10M"]):
    from datetime import datetime
    date_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
    for corpus_total_token_num_choice in corpus_total_token_num_choices:
        for corpus_split in CORPUS_SPLITS:
            for set_split in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS:
                filepaths = get_set_split_valid_filepaths(os.path.join(general_data_dir, corpus_total_token_num_choice, corpus_split), set_split)
                for filepath in filepaths:
                    prefix = filepath.split(".")[0]
                    basename = os.path.basename(prefix)
                    shutil.copy(src=filepath, dst=os.path.join(backup_data_dir,f"{basename}_{corpus_total_token_num_choice}_{corpus_split}_legacy_{date_suffix}.jsonl"))

def replace_legacy_leaner(general_data_dir, corpus_total_token_num_choices=["10M"]):
    backup_already = input(f"Have you made backup copy of the files under [{general_data_dir}] to be removed? [y/n]:")
    if not (backup_already == "y" or backup_already == ""):
        exit()
    for corpus_total_token_num_choice in corpus_total_token_num_choices:
        for corpus_split in CORPUS_SPLITS:
            for set_split in NATURAL_LANGUAGE_SET_SPLITS + FORMAL_LANGUAGE_SET_SPLITS:
                filepaths = get_set_split_valid_filepaths(os.path.join(general_data_dir, corpus_total_token_num_choice, corpus_split), set_split)
                for filepath in filepaths:
                    prefix = filepath.split(".")[0]
                    replacement_filepath = prefix + "_processed.jsonl"
                    assert os.path.exists(replacement_filepath)
                    os.remove(filepath)
                    shutil.move(src=replacement_filepath, dst=filepath)

def remove_duplicated_backup_leaner(backup_data_dir, date_suffix_list):
    for filename in os.listdir(backup_data_dir):
        filepath = os.path.join(backup_data_dir, filename)
        if os.path.isfile(filepath) and any(date_suffix in filepath for date_suffix in date_suffix_list):
            os.remove(filepath)

def restore_legacy_leaner(general_data_dir, backup_data_dir, date_suffix):
    for filename in os.listdir(backup_data_dir):
        filepath = os.path.join(backup_data_dir, filename)
        if os.path.isfile(filepath) and date_suffix in filepath:
            prefix = filepath.split(".")[0]
            basename = os.path.basename(prefix)
            leaner_name, corpus_total_token_num_choice, corpus_split, _, _ = basename.split("_")
            replacement_filepath = os.path.join(general_data_dir, corpus_total_token_num_choice, corpus_split, f"{leaner_name}.jsonl")
            assert os.path.isfile(replacement_filepath)
            os.remove(replacement_filepath)
            shutil.copy(src=filepath, dst=replacement_filepath)

def post_process_leaner(general_data_dir, backup_data_dir, corpus_total_token_num_choices=["100M"]):
    make_leaner_copy(general_data_dir=general_data_dir, backup_data_dir=backup_data_dir)
    process_leaner(general_data_dir=general_data_dir, action=action_expression_replacement_adapt, corpus_total_token_num_choices=corpus_total_token_num_choices)
    replace_legacy_leaner(general_data_dir=general_data_dir, corpus_total_token_num_choices=corpus_total_token_num_choices)
    deduplicate_leaner(general_data_dir=general_data_dir, corpus_total_token_num_choices=corpus_total_token_num_choices)
    replace_legacy_leaner(general_data_dir=general_data_dir, corpus_total_token_num_choices=corpus_total_token_num_choices)

def post_process_leaner_instruction(general_data_dir, backup_data_dir):
    from datetime import datetime
    date_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
    right_filepath = os.path.join(general_data_dir, "instruction0000.jsonl")
    replacement_filepath = os.path.join(general_data_dir, "instruction0000_processed.jsonl")
    legacy_filepath = os.path.join(backup_data_dir, f"instruction0000_legacy_{date_suffix}.jsonl")
    shutil.copy(src=right_filepath, dst=legacy_filepath)
    process_file_in_leaner(filepath=right_filepath, set_split="instruction", action=action_expression_replacement_adapt)
    os.remove(right_filepath)
    shutil.move(src=replacement_filepath, dst=right_filepath)
    deduplicate_file_in_leaner(filepath=right_filepath)
    os.remove(right_filepath)
    shutil.move(src=replacement_filepath, dst=right_filepath)

def get_leaner_filename_dict(leaner_dir, set_splits):
    if not os.path.exists(os.path.join(leaner_dir, "train")):
        return {"train": [filename for filename in os.listdir(leaner_dir) if re.sub(r'\d.*', '', filename) in set_splits]}
    filename_dict = {}
    for split in SPLIT_COMPOSITION:
        if os.path.exists(os.path.join(leaner_dir, split)):
            filename_dict[split] = [filename for filename in os.listdir(os.path.join(leaner_dir, split)) if re.sub(r'\d.*', '', filename) in set_splits]
        else:
            filename_dict[split] = None
    return filename_dict

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def load_txt(file_path):
    data = open(file_path, "r").readlines()
    data = [d.strip() for d in data]
    return data

def process_paragraphs(template, paragraphs, sent_num=2, token_threshold=800):
    def parse_edited(text):
        pattern = r'EDITED:(.*)$'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            edited_text = match.group(1).strip()
            return edited_text
        else:
            return ""
    sentence_list = []
    join_char_list = []
    for p in paragraphs:
        sentences = tokenize_sentences(p)
        sentences = [t for s in sentences for t in s.split("\n") if t != ""]
        sentence_list += sentences
        reduced_p = deepcopy(p)
        join_char_list += [" <end>"]
        for i in range(len(sentences[:-1])):
            sentence_i_stop = reduced_p.find(sentences[i])+len(sentences[i])
            sentence_i_1_start = reduced_p.find(sentences[i+1])
            join_char_list += [reduced_p[sentence_i_stop:sentence_i_1_start]]
            reduced_p = reduced_p[sentence_i_1_start:]
    join_char_list[0] = ""
    partial_paragraph = []
    start_idx = 0
    processed_paragraph = ""
    for i in range(0, len(sentence_list), sent_num):
        partial_paragraph += sentence_list[i:i+sent_num]
        if i + sent_num >= len(sentence_list):
            join_char_start_idx = join_char_list[start_idx]
            join_char_list[start_idx] = ""
            content = "".join([join_char_list[j] + sentence_list[j] for j in range(start_idx, start_idx+len(partial_paragraph))]) + " <end>"
            try:
                res = get_gpt_response(template.replace("{paragraph}", content))
                res = parse_edited(res)
                assert res.endswith("<end>")
                processed_paragraph += join_char_start_idx + res[:-len("<end>")].strip()
            except Exception as e:
                if "exceed" in str(e):
                    logging.error(str(e))
                    raise NotImplementedError()
                if 'content' in locals():
                    logging.error(str(e) + f" {repr(content)}")
                else:
                    logging.error(str(e))
                # processed_paragraph += " <error_start>{}<error_end>".format(join_char_start_idx + content[:-len("<end>")].strip())
        elif len(tokenizer.tokenize("\n".join(partial_paragraph))) > token_threshold:
            join_char_start_idx = join_char_list[start_idx]
            join_char_list[start_idx] = ""
            content = "".join([join_char_list[j] + sentence_list[j] for j in range(start_idx, start_idx+len(partial_paragraph))]) + " <end>"
            partial_paragraph = []
            start_idx = i + sent_num
            try:
                res = get_gpt_response(template.replace("{paragraph}", content))
                res = parse_edited(res)
                assert res.endswith("<end>")
                processed_paragraph += join_char_start_idx + res[:-len("<end>")].strip()
            except Exception as e:
                if "exceed" in str(e):
                    logging.error(str(e))
                    raise NotImplementedError()
                if 'content' in locals():
                    logging.error(str(e) + f" {repr(content)}")
                else:
                    logging.error(str(e))
                # processed_paragraph += " <error_start>{}<error_end>".format(join_char_start_idx + content[:-len("<end>")].strip())
    return processed_paragraph.split(" <end>")

def process_paragraphs_code(template, paragraphs):
    processed_paragraphs = []
    for p in paragraphs:
        task, response = p[0].strip(), p[1].strip()
        try:
            res = get_gpt_response(template.replace("{task}", task).replace("{response}", response))
            res_dict = parse_elements(text=res, key_list=["edited task description", "edited solution and code"])
            assert res_dict["edited task description"].endswith("<end>") and res_dict["edited solution and code"].endswith("<end>")
            _task, _response = res_dict["edited task description"][:-len("<end>")].strip(), res_dict["edited solution and code"][:-len("<end>")].strip()
            processed_paragraphs.append((_task, _response))
        except Exception as e:
            if 'res' in locals():
                logging.error(str(e) + f" {repr(res)}")
            elif "exceed" in str(e):
                logging.error(str(e))
                raise NotImplementedError()
            else:
                logging.error(str(e))
            processed_paragraphs.append((res, ""))
    return processed_paragraphs

def process_paragraphs_instruction(template, paragraphs):
    processed_paragraphs = []
    for p in paragraphs:
        task, input, response = p[0].strip(), p[1].strip(), p[2].strip()
        try:
            res = get_gpt_response(template.replace("{task}", task).replace("{input}", input).replace("{response}", response))
            assert ("<split>" in res or ("[INPUT]" in res and "[RESPONSE]" in res)) and res.endswith("<end>")
            if "<split>" in res:
                _edited = res.split("<split>")
            elif "[INPUT]" in res and "[RESPONSE]" in res:
                _edited = [j for m in res.split("[INPUT]") for j in m.split("[RESPONSE]")]
                if _edited[0].strip().startswith("[TASK]"):
                    _edited[0] = _edited[0].strip()[len("[TASK]"):]
            assert len(_edited) == 3
            _task, _input, _response = _edited[0].strip(), _edited[1].strip(), _edited[2][:-len("<end>")].strip()
            processed_paragraphs.append((_task, _input, _response))
        except Exception as e:
            logging.error(str(e))
            try:
                processed_paragraphs.append((res, "", ""))
            except Exception as e:
                logging.error(str(e))
                processed_paragraphs.append(None)
    return processed_paragraphs

def get_template(template_filepath):
    return "".join(open(template_filepath, "r").readlines())

def load_json_zst_file(filepath):
    with open(filepath, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = reader.read()
            data_list = json.loads(data.decode('utf-8'))
    return data_list

def load_jsonl_zst_file(filepath):
    data_list = []
    if filepath.endswith('.jsonl.zst'):
        with open(filepath, 'rb') as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as reader:
                buffer = b''
                while True:
                    chunk = reader.read(65536)  # Adjust the buffer size as needed
                    if not chunk:
                        break
                    buffer += chunk
                    lines = buffer.splitlines(True)
                    for line in lines[:-1]:
                        data_list.append(json.loads(line.decode('utf-8')))
                    buffer = lines[-1]
            data_list.append(json.loads(buffer.splitlines(True)[0].decode('utf-8')))
    return data_list

def sorted_set_splits(set_splits):
    new_set_splits = sorted(list(set(NATURAL_LANGUAGE_SET_SPLITS+FORMAL_LANGUAGE_SET_SPLITS) & set(set_splits)), key=lambda item:(CORPUS_COMPOSITION[item], item), reverse=True)
    if "instruction" in set_splits:
        new_set_splits += INSTRUCTION_SET_SPLITS
    return new_set_splits

def load_leaner_dataset(general_data_dir, set_splits=NATURAL_LANGUAGE_SET_SPLITS+FORMAL_LANGUAGE_SET_SPLITS, corpus_total_token_num_choice="100M", cache_dir=None):
    from datasets import load_dataset, DatasetDict, concatenate_datasets
    if not os.path.exists(os.path.join(general_data_dir, "10M")):
        dataset_list = []
        for set_split in set_splits:
            dataset = load_dataset("json", data_files=get_set_split_valid_filepaths(directory=general_data_dir, set_split=set_split), cache_dir=cache_dir)["train"]
            text_dataset = dataset.map(
                function=unify_set_split_text_field,
                batched=True
            )
            dataset_list.append(text_dataset)
        concatenated_dataset = concatenate_datasets(dataset_list).shuffle()
        return concatenated_dataset
    raw_datasets = DatasetDict({})
    for corpus_split in CORPUS_SPLITS:
        dataset_list = []
        if "instruction" in set_splits:
            if corpus_split == "train":
                dataset = load_dataset("json", data_files=[os.path.join(general_data_dir, "instruction0000.jsonl")], split='train[:99%]', cache_dir=cache_dir)
            elif corpus_split == "validation":
                dataset = load_dataset("json", data_files=[os.path.join(general_data_dir, "instruction0000.jsonl")], split='train[98%:99%]', cache_dir=cache_dir)
            elif corpus_split == "test":
                dataset = load_dataset("json", data_files=[os.path.join(general_data_dir, "instruction0000.jsonl")], split='train[99%:]', cache_dir=cache_dir)
            text_dataset = dataset.map(
                function=partial(unify_set_split_text_field, set_split="instruction"),
                batched=True
            )
            dataset_list.append(text_dataset)
        for set_split in sorted_set_splits(set_splits=set_splits):
            if set_split == "instruction":
                continue
            data_files = []
            for total_token_num_selection in TOTAL_TOKEN_NUM_FILEDIR_MAP[corpus_total_token_num_choice]:
                data_files += get_set_split_valid_filepaths(os.path.join(general_data_dir, total_token_num_selection, corpus_split), set_split=set_split)
            dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)["train"]
            text_dataset = dataset.map(
                function=partial(unify_set_split_text_field, set_split=set_split),
                batched=True
            )
            dataset_list.append(text_dataset)
        if dataset_list:
            concatenated_dataset = concatenate_datasets(dataset_list).shuffle()
            raw_datasets[corpus_split] = concatenated_dataset
    return raw_datasets

def load_leaner_dataset_with_retained_columns(general_data_dir, set_splits=NATURAL_LANGUAGE_SET_SPLITS+FORMAL_LANGUAGE_SET_SPLITS, corpus_total_token_num_choice="10M", retained_columns=["text", "set_split"]):
    dataset = load_leaner_dataset(general_data_dir=general_data_dir, set_splits=set_splits, corpus_total_token_num_choice=corpus_total_token_num_choice)
    dataset = dataset.remove_columns([k for k in set([column_name for k in dataset.column_names.keys() for column_name in dataset.column_names[k]]) if k not in retained_columns])
    return dataset

def load_original_set_split(filepaths, require_text_for_hash=False):
    from datasets import Dataset
    dataset_list = []
    for filepath in filepaths:
        dataset_list += load_jsonl_zst_file(filepath)
    dataset = Dataset.from_list(dataset_list)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    if require_text_for_hash:
        dataset = dataset.add_column('text_for_hash', dataset['text'])
    return dataset

def load_original_wiki(filepaths, require_text_for_hash=False):
    from datasets import Dataset
    dataset_list = []
    for filepath in filepaths:
        dataset_list += load_jsonl_zst_file(filepath)
    dataset_list = [{"text": list(dataset_element.values())[0]} for dataset_element in dataset_list]
    dataset = Dataset.from_list(dataset_list)
    if require_text_for_hash:
        dataset = dataset.add_column('text_for_hash', dataset['text'])
    return dataset

def load_original_textbook(filepaths, require_text_for_hash=False):
    from datasets import Dataset
    dataset_list = []
    for filepath in filepaths:
        dataset_list += load_jsonl_zst_file(filepath)
    dataset = Dataset.from_list(dataset_list)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "textbook"])
    dataset = dataset.rename_column("textbook", "text")
    if require_text_for_hash:
        dataset = dataset.add_column('text_for_hash', dataset['text'])
    return dataset

def load_original_conversation(filepaths, require_text_for_hash=False):
    from datasets import Dataset
    dataset_list = []
    for filepath in filepaths:
        dataset_list += load_json_zst_file(filepath)
    dataset_list = [{"text": dataset_element} for dataset_element in dataset_list]
    dataset = Dataset.from_list(dataset_list)
    if require_text_for_hash:
        dataset = dataset.add_column('text_for_hash', dataset['text'])
    return dataset

def load_original_code(filepaths, require_text_for_hash=False):
    from datasets import Dataset
    dataset_list = []
    for filepath in filepaths:
        dataset_list += load_jsonl_zst_file(filepath)
    dataset = Dataset.from_list(dataset_list)
    dataset = dataset.map(
        function=partial(unify_set_split_text_field, set_split="code", require_text_for_hash=require_text_for_hash), 
        batched=True
    )
    if require_text_for_hash:
        remained_cols = ["text", "text_for_hash"]
    else:
        remained_cols = ["text"]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in remained_cols])
    return dataset

def load_original_math(filepaths, require_text_for_hash=False):
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=filepaths)["train"]
    dataset = dataset.map(
        function=partial(unify_set_split_text_field, set_split="math", require_text_for_hash=require_text_for_hash),
        batched=True
    )
    if require_text_for_hash:
        remained_cols = ["text", "text_for_hash"]
    else:
        remained_cols = ["text"]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in remained_cols])
    return dataset

def load_original_instruction(corpus_split="train", require_text_for_hash=False, cache_dir=None):
    from datasets import load_dataset
    if corpus_split == "train":
        dataset = load_dataset("yahma/alpaca-cleaned", split='train[:98%]', cache_dir=cache_dir)
    elif corpus_split == "validation":
        dataset = load_dataset("yahma/alpaca-cleaned", split='train[98%:99%]', cache_dir=cache_dir)
    elif corpus_split == "test":
        dataset = load_dataset("yahma/alpaca-cleaned", split='train[99%:]', cache_dir=cache_dir)
    dataset = dataset.map(
        function=partial(unify_set_split_text_field, set_split="instruction", require_text_for_hash=require_text_for_hash),
        batched=True
    )
    if require_text_for_hash:
        remained_cols = ["text", "text_for_hash"]
    else:
        remained_cols = ["text"]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in remained_cols])
    return dataset

SET_SPLIT_ORIGINAL_DATASET_MAP = {
    "web": load_original_set_split,
    "book": load_original_set_split,
    "wiki": load_original_wiki,
    "textbook": load_original_textbook,
    "conversation": load_original_conversation,
    "code": load_original_code,
    "math": load_original_math,
    "instruction": load_original_instruction
}

def load_original_dataset(general_data_dir, set_splits=NATURAL_LANGUAGE_SET_SPLITS+FORMAL_LANGUAGE_SET_SPLITS, corpus_total_token_num_choice="10M", require_text_for_hash=False, cache_dir=None):
    from datasets import DatasetDict, concatenate_datasets
    if not os.path.exists(os.path.join(general_data_dir, "10M")):
        raise NotImplementedError()
    raw_datasets = DatasetDict({})
    for corpus_split in CORPUS_SPLITS:
        dataset_list = []
        if "instruction" in set_splits:
            dataset = load_original_instruction(corpus_split, require_text_for_hash, cache_dir=cache_dir)
            dataset_list.append(dataset)
        for set_split in sorted_set_splits(set_splits=set_splits):
            if set_split == "instruction":
                continue
            data_files = []
            for total_token_num_selection in TOTAL_TOKEN_NUM_FILEDIR_MAP[corpus_total_token_num_choice]:
                data_files += get_original_set_split_valid_filepaths(os.path.join(general_data_dir, total_token_num_selection, corpus_split), set_split=set_split)
            dataset = SET_SPLIT_ORIGINAL_DATASET_MAP[set_split](data_files, require_text_for_hash)
            dataset_list.append(dataset)
        if dataset_list:
            concatenated_dataset = concatenate_datasets(dataset_list).shuffle()
            raw_datasets[corpus_split] = concatenated_dataset
    return raw_datasets

def get_hash_set_and_token_num(directory, set_split):
    hash_value = ""
    hash_set = set()
    token_num = 0
    for filepath in get_set_split_valid_filepaths(directory, set_split):
        print(filepath)
        with open(filepath, 'r') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    hash_value = json_obj.get("hash")
                    if hash_value is not None:
                        hash_set.add(hash_value)
                    if set_split == "instruction":
                        token_num += count_tokens_in_text(tokenizer=tokenizer, text=" ".join([json_obj[tf] for tf in ["instruction", "input", "output"] if json_obj[tf] != ""]))
                    elif set_split == "code":
                        token_num += count_tokens_in_text(tokenizer=tokenizer, text=" ".join([json_obj[tf] for tf in ["prompt", "response"] if json_obj[tf] != ""]))
                    else:
                        token_num += count_tokens_in_text(tokenizer=tokenizer, text=json_obj.get("text"))
                except json.JSONDecodeError:
                    pass      
    last_hash = hash_value          
    return last_hash, hash_set, token_num

def get_set_split_valid_filepaths(directory, set_split):
    if not os.path.exists(directory):
        return []
    idxs = [int(filename.split(".")[0][len(set_split):]) for filename in os.listdir(directory) if is_valid_leaner_filename(filename) and filename.startswith(set_split)]
    idxs = sorted(idxs)
    filepaths = [os.path.join(directory, "{}{:04d}.jsonl".format(set_split, idx)) for idx in idxs]
    return filepaths

def get_original_set_split_valid_filepaths(directory, set_split):
    if not os.path.exists(directory):
        return []
    filenames = [filename for filename in os.listdir(directory) if is_valid_original_filename(filename) and filename.startswith(set_split)]
    if not filenames:
        return []
    idxs = [int(filename.split(".")[0][len(set_split):]) for filename in filenames]
    idxs = sorted(idxs)
    suffix = ".".join(filenames[0].split(".")[1:])
    filepaths = [os.path.join(directory, "{}{:04d}.{}".format(set_split, idx, suffix)) for idx in idxs]
    return filepaths

def get_set_split_idx_and_token_num_buffer(directory, set_split):
    idxs = [int(filename.split(".")[0][len(set_split):]) for filename in os.listdir(directory) if is_valid_leaner_filename(filename) and set_split in filename]
    if len(idxs) > 0:
        idx = sorted(idxs, reverse=True)[0]
        token_num_buffer = count_tokens_in_leaner_file(filepath=os.path.join(directory, "{}{:04d}.jsonl".format(set_split, idx)), set_split=set_split)
        return idx, token_num_buffer
    else:
        return 0, 0

def yield_samples_in_batches(sample_generator, batch_size):
    batch = []
    for sample in sample_generator:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def is_low_quality_article(article_content):
    pattern = r'\b\d+\b|\b(?:https?://)?(?:www\.)?[#a-zA-Z0-9-]+\.(?:com|net|org|io|gov|edu)\b'

    matches = re.findall(pattern, article_content)

    if len(matches) / len(article_content.split()) > 0.1:
        return True
    else:
        return False

def remove_error_pattern(text):
    entire_pattern = re.compile(r'<error_start>.*?<error_end>', re.DOTALL)
    start_pattern = re.compile(r'<error_start>.*')
    end_pattern = re.compile(r'.*<error_end>')
    return end_pattern.sub("", start_pattern.sub("", entire_pattern.sub("", text)))

def remove_angle_brackets(text):
    pattern = r"<[^>]*>"
    return re.sub(pattern, "", text)

def remove_non_ascii_characters(text):
    replacements = {
        '\u2013': '-',   # en-dash
        '\u2014': '--',  # em-dash
        '\u00bb': '>>',  # right-pointing double angle quotation mark
        '\u2026': '...', # horizontal ellipsis
        '\u2190': '<-',  # leftwards arrow
        '\u2192': '->',  # rightwards arrow
        '\u2022': '-',   # bullet
        '\u00ab': '<<',  # left-pointing double angle quotation mark
        '\u00b7': '*',   # middle dot
    }
    clean_text = deepcopy(text)
    for old, new in replacements.items():
        clean_text = clean_text.replace(old, new)
    pattern = r'[^\x00-\x7F]'
    clean_text = re.sub(pattern, '', clean_text)
    return clean_text

def slimpajama_processor(d):
    return d

def wiki_processor(d):
    for k in d.keys():
        buffer = {
            "idx": int(k),
            "text": remove_angle_brackets(d[k]).strip()
        }
    return buffer

def textbook_processor(d):
    buffer = deepcopy(d)
    buffer["text"] = d["textbook"]
    buffer.pop("textbook")
    buffer["original text"] = d["text"]
    return buffer

def conversation_processor(d):
    buffer = {
        "text": d
    }
    return buffer

def code_processor(d):
    d["text"] = d["prompt"] + " " + d["response"]
    return d

SET_SPLIT_PROCESSOR_MAP = {
    "web": slimpajama_processor,
    "book": slimpajama_processor,
    "wiki": wiki_processor,
    "textbook": textbook_processor,
    "conversation": conversation_processor,
    "code": code_processor,
}

def ori_jsonl_zst_generator(directory, set_split, hash_set, last_hash, dismiss_principle="last"):
    hit_last = False if last_hash != "" else True
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(set_split)]
    filepaths = sorted(filepaths)
    for filepath in filepaths:
        if os.path.isfile(filepath):
            print(f"\t{filepath}")
            dataset = load_jsonl_zst_file(filepath)
            for d in dataset:
                processed_d = SET_SPLIT_PROCESSOR_MAP[set_split](d)
                if dismiss_principle == "last" and not hit_last and fixed_hash(processed_d["text"]) != last_hash:
                    continue
                elif dismiss_principle == "last" and not hit_last and fixed_hash(processed_d["text"]) == last_hash:
                    hit_last = True
                    continue
                if dismiss_principle == "set" and fixed_hash(processed_d["text"]) in hash_set:
                    continue
                yield SET_SPLIT_PROCESSOR_MAP[set_split](d)

def ori_json_zst_generator(directory, set_split, hash_set, last_hash, dismiss_principle="last"):
    print(dismiss_principle)
    hit_last = False if last_hash != "" else True
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(set_split)]
    filepaths = sorted(filepaths)
    for filepath in filepaths:
        if os.path.isfile(filepath):
            dataset = load_json_zst_file(filepath)
            for d in dataset:
                processed_d = SET_SPLIT_PROCESSOR_MAP[set_split](d)
                if dismiss_principle == "last" and not hit_last and fixed_hash(processed_d["text"]) != last_hash:
                    continue
                elif dismiss_principle == "last" and not hit_last and fixed_hash(processed_d["text"]) == last_hash:
                    hit_last = True
                if dismiss_principle == "set" and fixed_hash(processed_d["text"]) in hash_set:
                    continue
                yield SET_SPLIT_PROCESSOR_MAP[set_split](d)

def ori_dataset_generator(name, cache_dir, text_fields, hash_set, last_hash):
    from datasets import load_dataset
    dataset = load_dataset(name, cache_dir=cache_dir)["train"]
    for d in dataset:
        text = [d[tf] for tf in text_fields if d[tf] != None and d[tf] != ""]
        text = " ".join(text)
        if "text" in d.keys():
            d["original text"] = d["text"]
        d["text"] = text
        if fixed_hash(text) not in hash_set:
            yield d

SET_SPLIT_BATCH_SIZE_MAP = {
    "web": 1,
    "book": 1,
    "wiki": 3,
    "textbook": 2,
    "conversation": 1,
    "code": 1,
}

SET_SPLIT_SENT_NUM_MAP = {
    "web": 1,
    "book": 1,
    "wiki": 2,
    "textbook": 2,
    "conversation": 2,
    "code": None,
}

SET_SPLIT_TOKEN_THRESHOLD_MAP = {
    "web": 300,
    "book": 300,
    "wiki": 300,
    "textbook": 600,
    "conversation": 600,
    "code": None,
}

SET_SPLIT_ORIGINAL_GENERATOR_MAP = {
    "web": ori_jsonl_zst_generator,
    "book": ori_jsonl_zst_generator,
    "wiki": ori_jsonl_zst_generator,
    "textbook": ori_jsonl_zst_generator,
    "conversation": ori_json_zst_generator,
    "code": ori_jsonl_zst_generator,
    "instruction": ori_dataset_generator,
}

def process_set_split(template_filepath, general_file_dir, general_output_dir, target_token_num_choices=["10M", "100M", "1B", "10B"], set_split="web", debug=False):
    print(f"Processing {set_split}...")
    print(f"Output directory: {general_output_dir}")
    template = get_template(template_filepath)
    for target_token_num_choice in target_token_num_choices:
        file_dir = os.path.join(general_file_dir, target_token_num_choice)
        output_dir = os.path.join(general_output_dir, target_token_num_choice)
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(os.path.join(output_dir, 'error.log')), level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        for corpus_split in CORPUS_SPLITS:
            os.makedirs(os.path.join(output_dir, corpus_split), exist_ok=True)
        batch_size = SET_SPLIT_BATCH_SIZE_MAP[set_split]
        token_num = 0
        for corpus_split in CORPUS_SPLITS:
            last_hash, hash_set, token_num = get_hash_set_and_token_num(os.path.join(output_dir, corpus_split), set_split)
            set_split_idx, token_num_buffer = get_set_split_idx_and_token_num_buffer(os.path.join(output_dir, corpus_split), set_split)
            print(f"Processing {corpus_split}")
            print(f"Processed sample number: {len(hash_set)} with {token_num} tokens")
            print(f"Starting from idx {set_split_idx} with {token_num_buffer} token buffer")
            for batch in yield_samples_in_batches(SET_SPLIT_ORIGINAL_GENERATOR_MAP[set_split](os.path.join(file_dir, corpus_split), set_split=set_split, hash_set=hash_set, last_hash=last_hash, dismiss_principle="set"), batch_size=batch_size):
                if set_split != "code":
                    paragraphs = [remove_non_ascii_characters(d["text"]) for d in batch]
                    new_paragraphs = process_paragraphs(template=template, paragraphs=paragraphs, sent_num=SET_SPLIT_SENT_NUM_MAP[set_split], token_threshold=SET_SPLIT_TOKEN_THRESHOLD_MAP[set_split])
                else:
                    paragraphs = [(d["prompt"], d["response"]) for d in batch]
                    new_paragraphs = process_paragraphs_code(template=template, paragraphs=paragraphs)
                
                for new_paragraph, d in zip(new_paragraphs, batch):
                    if set_split != "code":
                        new_paragraph = remove_non_ascii_characters(remove_error_pattern(new_paragraph)).strip()
                        if new_paragraph != "" and len(new_paragraph) > 100: 
                            d["hash"] = fixed_hash(d["text"])
                            d["text"] = new_paragraph
                            tokens_in_text = count_tokens_in_text(tokenizer=tokenizer, text=new_paragraph)
                            token_num += tokens_in_text
                            token_num_buffer += tokens_in_text
                            if token_num_buffer > 10e6:
                                token_num_buffer = 0
                                set_split_idx += 1
                            with open(os.path.join(output_dir, corpus_split, "{}{:04d}.jsonl".format(set_split, set_split_idx)), "a") as af:
                                af.write(f"{json.dumps(d)}\n")
                    else:
                        if new_paragraph and new_paragraph[1] != "":
                            d["hash"] = fixed_hash(d.pop("text"))
                            d["prompt"] = new_paragraph[0]
                            d["response"] = new_paragraph[1]
                            tokens_in_text = count_tokens_in_text(tokenizer=tokenizer, text=new_paragraph[0]+" "+new_paragraph[1])
                            token_num += tokens_in_text
                            token_num_buffer += tokens_in_text
                            if token_num_buffer > 10e6:
                                token_num_buffer = 0
                                set_split_idx += 1
                            with open(os.path.join(output_dir, corpus_split, "{}{:04d}.jsonl".format(set_split, set_split_idx)), "a") as af:
                                af.write(f"{json.dumps(d)}\n")
                    
                if debug and token_num > 1e3:
                    return
                
                time.sleep(2)
            with open(os.path.join(output_dir, "statistics.txt"), "a") as af:
                renewed_token_num = count_tokens_in_leaner(os.path.join(output_dir, corpus_split), set_split)
                af.write("{}/{}: {} ({:.4f} * 100M) tokens\n".format(corpus_split, set_split, renewed_token_num, renewed_token_num / 100e6))

def _process_instruction(template_filepath, general_output_dir, debug=False):
    set_split = "instruction"
    print(f"Processing instruction...")
    print(f"Output directory: {general_output_dir}")
    template = get_template(template_filepath)

    os.makedirs(general_output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(os.path.join(general_output_dir, 'error.log')), level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    batch_size = 1
    token_num = 0

    last_hash, hash_set, token_num = get_hash_set_and_token_num(os.path.join(general_output_dir), set_split)
    set_split_idx, token_num_buffer = get_set_split_idx_and_token_num_buffer(os.path.join(general_output_dir), set_split)

    print(f"Processed sample number: {len(hash_set)} with {token_num} tokens")
    print(f"Starting from idx {set_split_idx} with {token_num_buffer} token buffer")
    for batch in yield_samples_in_batches(SET_SPLIT_ORIGINAL_GENERATOR_MAP[set_split]("yahma/alpaca-cleaned", cache_dir=CACHE_DIR, text_fields=["instruction", "input", "output"], hash_set=hash_set), batch_size=batch_size):
        paragraphs = [(d["instruction"], d["input"], d["output"]) for d in batch]
        new_paragraphs = process_paragraphs_instruction(template=template, paragraphs=paragraphs)
        for new_paragraph, d in zip(new_paragraphs, batch):
            if new_paragraph:
                d["hash"] = fixed_hash(d.pop("text"))
                d["instruction"] = new_paragraph[0]
                if new_paragraph[1] == "None":
                    d["input"] = ""
                else:
                    d["input"] = new_paragraph[1]
                d["output"] = new_paragraph[2]
                tokens_in_text = count_tokens_in_text(tokenizer=tokenizer, text=" ".join([d[tf] for tf in ["instruction", "input", "output"] if d[tf] != None and d[tf] != ""]))
                token_num += tokens_in_text
                token_num_buffer += tokens_in_text
                if token_num_buffer > 10e6:
                    token_num_buffer = 0
                    set_split_idx += 1
                with open(os.path.join(general_output_dir, "{}{:04d}.jsonl".format(set_split, set_split_idx)), "a") as af:
                    af.write(f"{json.dumps(d)}\n")
        time.sleep(2)
        if debug and token_num > 1e2:
            return
    with open(os.path.join(general_output_dir, "statistics.txt"), "a") as af:
        renewed_token_num = count_tokens_in_leaner(general_output_dir, set_split)
        af.write("{}: {} ({:.4f} * 100M) tokens\n".format(set_split, renewed_token_num, renewed_token_num / 100e6))

def process_web(debug=False):
    target_token_num_choices = ["10M", "100M"]
    set_split = "web"
    template_filepath = os.path.join(TEMPLATE_DIR, "web_simplification.txt")
    general_file_dir = os.path.join(DATA_DIR, "ori")
    if not debug:
        general_output_dir = DATA_DIR
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split)
    else:
        general_output_dir = os.path.join(DATA_DIR, "web")
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split, debug=True)      

def process_book(debug=False):
    target_token_num_choices = ["10M", "100M"]
    set_split = "book"
    template_filepath = os.path.join(TEMPLATE_DIR, "book_simplification.txt")
    general_file_dir = os.path.join(DATA_DIR, "ori")
    if not debug:
        general_output_dir = DATA_DIR
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split)
    else:
        general_output_dir = os.path.join(DATA_DIR, "book")
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split, debug=True)

def process_wiki(debug=False):
    target_token_num_choices = ["10M", "100M"]
    set_split = "wiki"
    template_filepath = os.path.join(TEMPLATE_DIR, "wiki_simplification.txt")
    general_file_dir = os.path.join(DATA_DIR, "ori")
    if not debug:
        general_output_dir = DATA_DIR
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split)
    else:
        general_output_dir = os.path.join(DATA_DIR, "wiki")
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split, debug=True)

def process_textbook(debug=False):
    target_token_num_choices = ["10M", "100M"]
    set_split = "textbook"
    template_filepath = os.path.join(TEMPLATE_DIR, "textbook_simplification.txt")
    general_file_dir = os.path.join(DATA_DIR, "ori")
    if not debug:
        general_output_dir = DATA_DIR
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split)
    else:
        general_output_dir = os.path.join(DATA_DIR, "textbook")
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split, debug=True)

def process_conversation(debug=False):
    target_token_num_choices = ["10M", "100M"]
    set_split = "conversation"
    template_filepath = os.path.join(TEMPLATE_DIR, "conversation_simplification.txt")
    general_file_dir = os.path.join(DATA_DIR, "ori")
    if not debug:
        general_output_dir = DATA_DIR
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split)
    else:
        general_output_dir = os.path.join(DATA_DIR, "conversation")
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split, debug=True)

def process_code(debug=False):
    target_token_num_choices = ["10M", "100M"]
    set_split = "code"
    template_filepath = os.path.join(TEMPLATE_DIR, "code_simplification.txt")
    general_file_dir = os.path.join(DATA_DIR, "ori")
    if not debug:
        general_output_dir = DATA_DIR
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split)
    else:
        general_output_dir = os.path.join(DATA_DIR, "code")
        process_set_split(template_filepath=template_filepath, general_file_dir=general_file_dir, general_output_dir=general_output_dir, target_token_num_choices=target_token_num_choices, set_split=set_split, debug=True)

def process_instruction(debug=False):
    template_filepath = os.path.join(TEMPLATE_DIR, "instruction_simplification.txt")
    if not debug:
        general_output_dir = DATA_DIR
        _process_instruction(template_filepath=template_filepath, general_output_dir=general_output_dir)
    else:
        general_output_dir = os.path.join(DATA_DIR, "instruction")
        _process_instruction(template_filepath=template_filepath, general_output_dir=general_output_dir, debug=True)

def renew_dataset(model, tokenizer, dataset, corpus_splits=["train"], save=True, output_data_dir=None, epoch=None, store_text=False):
    from datasets import Dataset
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if isinstance(dataset, Dataset):
        dataset = dataset.map(
        partial(action_compute_loss, model=model, tokenizer=tokenizer, set_split="N/A", save=save)
        )
        if save:
            os.makedirs(os.path.join(output_data_dir), exist_ok=True)
            filename = "set_split_loss_epoch{:04d}.jsonl".format(epoch) if epoch != None else "set_split_loss.jsonl"
            dataset.map(
                partial(action_store_loss, output_filepath=os.path.join(output_data_dir, filename), store_text=store_text)
            )
    else:
        for corpus_split in corpus_splits:
            dataset[corpus_split] = dataset[corpus_split].map(
            partial(action_compute_loss, model=model, tokenizer=tokenizer, set_split="N/A", save=save)
            )
            if save:
                os.makedirs(os.path.join(output_data_dir, corpus_split), exist_ok=True)
                filename = "set_split_loss_epoch{:04d}.jsonl".format(epoch) if epoch != None else "set_split_loss.jsonl"
                dataset[corpus_split].map(
                    partial(action_store_loss, output_filepath=os.path.join(output_data_dir, corpus_split, filename), store_text=store_text)
                )
    return dataset

def process_dataset_set_split_loss(dataset, epoch, output_dir, plot=True):
    if "token_num" not in dataset.features:
        dataset = dataset.map(
            function=estimate_token_num,
            batched=True
        )
    if "hist" not in dataset.features:
        dataset = dataset.map(
            function=estimate_hist,
            batched=True
        )
    df = pd.DataFrame(dataset)
    set_split_loss_dict = {}
    for set_split in df["set_split"].unique():
        selected_df = df[df["set_split"]==set_split]
        set_split_loss = (selected_df["loss"] * selected_df["token_num"]).sum() / selected_df["token_num"].sum()
        set_split_loss_dict[set_split] = set_split_loss
    if plot:
        figure_dir = os.path.join(output_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)
        sorted_data_key = sorted_set_splits(list(df["set_split"].unique()))
        # data = [set_split_loss_dict[k] for k in sorted_data_key]
        colors = [SET_SPLIT_COLOR_MAP[k] for k in sorted_data_key]
        for c, l in zip(colors, sorted_data_key):
            selected_df = df[df["set_split"]==l]
            hist = np.array([v for v in selected_df["hist"].values]).sum(axis=0)
            _, bin_edges = np.histogram(NUM_BINS, bins=NUM_BINS, range=LOSS_RANGE)
            pruned_hist = hist[:PLOT_NUM_BINS]
            pruned_bin_edges = bin_edges[:PLOT_NUM_BINS]
            # plt.hist(hists, bins=20, color=c, alpha=0.8, label=l)
            avr = set_split_loss_dict[l]
            plt.bar(pruned_bin_edges, pruned_hist, width=np.diff(bin_edges[:PLOT_NUM_BINS+1]), color=c, alpha=0.8, label=f"{l}: {avr:.2f}")
            plt.axvline(x=avr, color=c, linestyle='-')
        plt.legend(loc='upper right')
        plt.xlim(LOSS_RANGE[0], bin_edges[PLOT_NUM_BINS])
        plt.ylim(0, 100000)
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title(f"Epoch {epoch} Train Loss Dist.")  
        plt.savefig(os.path.join(figure_dir, "set_split_loss_epoch{:04d}.png".format(epoch)), dpi=256)
        plt.close()
    return set_split_loss_dict

def sample_ori_leaner_twins(original_data_dir, leaner_data_dir, set_split="random", corpus_split="train"):
    if set_split != "random":
        original_datasets = load_original_dataset(general_data_dir=original_data_dir, set_splits=[set_split], require_text_for_hash=True)
        leaner_datasets = load_leaner_dataset(general_data_dir=leaner_data_dir, set_splits=[set_split])
        if corpus_split != "random":
            original_dataset = original_datasets[corpus_split]
            leaner_dataset = leaner_datasets[corpus_split]
            idxs = list(range(len(original_dataset)))
            random.shuffle(idxs)
            for i in idxs:
                text_for_hash = original_dataset[i]["text_for_hash"]
                hash = fixed_hash(text_for_hash)
                entry = next((item for item in leaner_dataset if item['hash'] == hash), None)
                if entry:
                    yield text_for_hash, entry["text"]

def average_sentence_tokens_spacy(nlp, text):
    doc = nlp(text)
    
    sentences = [sent.text for sent in doc.sents]
    
    sentence_lengths = [len(sent.split()) for sent in sentences]
    
    if sentence_lengths:
        return sum(sentence_lengths) / len(sentence_lengths)
    else:
        return 0

def compute_average_sentence_tokens(filepath):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10**7

    json_objs = load_jsonl(filepath)

    new_json_objs = []
    for json_obj in json_objs:
        json_obj["average_sentence_tokens"] = average_sentence_tokens_spacy(nlp, json_obj["text"])
        new_json_objs.append(json_obj)
    
    with open(filepath, "w") as wf:
        for json_obj in new_json_objs:
            wf.write(f"{json.dumps(json_obj)}\n")


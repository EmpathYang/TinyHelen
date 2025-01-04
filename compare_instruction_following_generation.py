import random
import os
import json
import re
import csv
import matplotlib.pyplot as plt

from utils import INSTRUCTION_FOLLOWING_QUESTIONS_DIR, TEMPLATE_DIR, GENERATION_DIR, GENERATION_EVALUATION_DIR, load_jsonl, get_template
from gpt import get_gpt_response

def sample_questions_for_inner_comparison(sources=["gpt4", "chatgpt"]):
    with open(os.path.join(INSTRUCTION_FOLLOWING_QUESTIONS_DIR, "inner_comparison.jsonl"), "w") as wf:
        for source in sources:
            question_filepath = os.path.join(INSTRUCTION_FOLLOWING_QUESTIONS_DIR, f"{source}.jsonl")
            questions = load_jsonl(question_filepath)
            sampled_questions = random.choices(questions, k=int(0.02 * len(questions)))
            for question in sampled_questions:
                wf.write(f"{json.dumps(question)}\n")

def get_model_generation_by_hash(model_id, hashes):
    model_generation_list = load_jsonl(os.path.join(GENERATION_DIR, f"{model_id}.jsonl"))
    model_generation_dict = {model_generation_json_obj["hash"]:model_generation_json_obj["response"] for model_generation_json_obj in model_generation_list}
    return [model_generation_dict[h] for h in hashes]

def get_question_and_hashes(question_filename):
    questions = load_jsonl(os.path.join(INSTRUCTION_FOLLOWING_QUESTIONS_DIR, question_filename))
    return [question["question"] for question in questions], [question["hash"] for question in questions]

def strip_generations(model_generations):
    clean_generations = []
    for generation in model_generations:
        clean_generation = generation.split("### RESPONSE:")[-1].strip()
        if clean_generation in ["", "None"]:
            clean_generation = "None (The model failed to generate anything, so simply rate it as the worst level.)"
        clean_generations.append(clean_generation)
    return clean_generations

def split_by_pattern(input_string):
    pattern = r'#*\s*[A-Z]\.\s*'
    result = re.split(pattern, input_string)
    result = [item.strip() for item in result if item.strip()]
    return result

def get_unfinished_buffer(comparison_task_name, questions, model_ids):
    finished_buffer = {}
    for model_id in model_ids:
        finished = load_jsonl(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name, f"{model_id}.jsonl"))
        finished_buffer[model_id] = [f["question"] for f in finished]
    unfinished_buffer = {}
    for question in questions:
        unfinished_buffer[question] = []
        for model_id in model_ids:
            if question not in finished_buffer[model_id]:
                unfinished_buffer[question].append(model_id)
    return unfinished_buffer

def compare_instruction_following_generation(comparison_task_name, model_ids, question_filename, chunk_size=10):
    os.makedirs(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name), exist_ok=True)
    questions, hashes = get_question_and_hashes(question_filename)
    model_generation_dict = {model_id: strip_generations(get_model_generation_by_hash(model_id, hashes)) for model_id in model_ids}
    instruction_following_generation_comparison_template = get_template(os.path.join(TEMPLATE_DIR, "instruction_following_generation_comparison.txt"))
    unfinished_buffer = get_unfinished_buffer(comparison_task_name, questions, model_ids)
    
    for i in range(len(questions)):
        question = questions[i]
        hash = hashes[i]
        for j in range(0, len(unfinished_buffer[question]), chunk_size):
            model_id_chunk = unfinished_buffer[question][j:j+chunk_size]
            random.shuffle(model_id_chunk)
            generations = "\n\n".join([f"###{chr(65 + (k % 26))}\n<response>" + model_generation_dict[model_id][i] + "</response>" for k, model_id in enumerate(model_id_chunk)])
            get_valid_evaluation = False
            attempts = 0
            while not get_valid_evaluation:
                attempts += 1
                if attempts == 3:
                    exit()
                get_valid_generation = False
                while not get_valid_generation:
                    prompt = instruction_following_generation_comparison_template.replace("{question}", question).replace("{generations}", generations)
                    gpt_response = get_gpt_response(prompt)
                    print(prompt)
                    print(gpt_response)
                    evaluations = split_by_pattern(gpt_response)
                    if len(evaluations) == len(model_id_chunk):
                        get_valid_generation = True
                if not (all(len(evaluation.split("\\t"))==3 for evaluation in evaluations) and all(e.isdigit() for evaluation in evaluations for e in evaluation.split("\\t"))):
                    continue
                else:
                    get_valid_evaluation = True
                for model_id, evaluation in zip(model_id_chunk, evaluations):
                    response = model_generation_dict[model_id][i]
                    grammar, coherence, specificity = evaluation.split("\\t")
                    grammar, coherence, specificity = int(grammar), int(coherence), int(specificity)
                    print(grammar, coherence, specificity)
                    evaluation_obj = {"question": question, "hash": hash, "response": response, "model": model_id, "evaluation": {"grammar": grammar, "coherence": coherence, "specificity": specificity}}
                    with open(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name, f"{model_id}.jsonl"), "a") as af:
                        af.write(f"{json.dumps(evaluation_obj)}\n")

def get_question_hashes(question_filenames):
    if type(question_filenames) == str:
        questions = load_jsonl(os.path.join(INSTRUCTION_FOLLOWING_QUESTIONS_DIR, question_filenames))
        return [question["hash"] for question in questions]
    else:
        questions = []
        for question_filename in question_filenames:
            questions += load_jsonl(os.path.join(INSTRUCTION_FOLLOWING_QUESTIONS_DIR, question_filename))
        return [question["hash"] for question in questions]

def get_data(comparison_task_name, model_ids, question_hashes=None):
    data = []
    for filename in [f"{model_id}.jsonl" for model_id in model_ids]:
        eval_data = load_jsonl(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name, filename))
        if question_hashes:
            eval_data = [q for q in eval_data if q["hash"] in question_hashes]
        data += eval_data
    return data

def get_top_models(data, metric, top_n=5):
    model_scores = {}

    for entry in data:
        model = entry['model']
        if not metric == "sum":
            score = entry['evaluation'].get(metric, 0)
        else:
            score = sum(list(entry["evaluation"].values()))
        
        if model not in model_scores:
            model_scores[model] = {'total': 0, 'count': 0}
        
        model_scores[model]['total'] += score
        model_scores[model]['count'] += 1
    
    for model, scores in model_scores.items():
        print(model, metric, scores['count'])
    average_scores = [(model, scores['total'] / scores['count']) for model, scores in model_scores.items()]
    
    sorted_models = sorted(average_scores, key=lambda x: x[1], reverse=True)
    
    top_models = sorted_models[:top_n]
    
    return top_models

def get_top_models_for_questions(data, metric, top_n=5):
    question_model_scores = {}

    for entry in data:
        question = entry['question']
        model = entry['model']
        if not metric == "sum":
            score = entry['evaluation'].get(metric, 0)
        else:
            score = sum(list(entry["evaluation"].values()))
        response = entry['response']
        
        if question not in question_model_scores:
            question_model_scores[question] = {}
        
        question_model_scores[question][model] = {'score': score, 'response': response}
            
    top_models_per_question = {}
    
    for question, models_scores in question_model_scores.items():
        average_scores = [(model, scores['score'], scores['response']) 
                          for model, scores in models_scores.items()]
        sorted_models = sorted(average_scores, key=lambda x: x[1], reverse=True)
        
        top_models_per_question[question] = {sorted_model[0]: sorted_model[1:] for sorted_model in sorted_models[:top_n]}
    
    return top_models_per_question

def inner_comparison():
    model_bases = [
        "llama-leaner-leaner-5e-3",
        "llama-leaner-ori-5e-3",
        "llama-ori-leaner-5e-3",
        "llama-ori-ori-5e-3",
    ]
    inner_comparison_model_ids = [f"{model_base}-{checkpoint}" for model_base in model_bases for checkpoint in range(100, 1001, 100)]
    compare_instruction_following_generation(comparison_task_name="inner_comparison", model_ids=inner_comparison_model_ids, question_filename="inner_comparison.jsonl")

def inter_comparison():
    inter_comparison_model_ids = ["llama-leaner-leaner-5e-3-700", "llama-leaner-ori-5e-3-600", "llama-ori-leaner-5e-3-800", "llama-ori-ori-5e-3-800"]
    compare_instruction_following_generation(comparison_task_name="inter_comparison", model_ids=inter_comparison_model_ids, question_filename="all.jsonl")

def evaluation_results(comparison_task_name, model_ids, question_hashes):
    data = get_data(comparison_task_name, model_ids, question_hashes)
    return {
        "sum": get_top_models(data, "sum"),
        "grammar": get_top_models(data, "grammar"),
        "coherence": get_top_models(data, "coherence"),
        "specificity": get_top_models(data, "specificity")
    }

def evaluation_jsonl_to_csv(comparison_task_name, filename, question_hashes=None):
    evaluation_filename = os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name, filename)

    fieldnames = ['question', 'grammar', 'coherence', 'specificity', 'hash', 'response', 'model']

    data = load_jsonl(evaluation_filename)
    if question_hashes:
        data = [q for q in data if q["hash"] in question_hashes]

    for item in data:
        item['evaluation_sum'] = item['evaluation']['grammar'] + item['evaluation']['coherence'] + item['evaluation']['specificity']

    data.sort(key=lambda x: x['evaluation_sum'], reverse=True)

    os.makedirs(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name+"_csv"), exist_ok=True)
    with open(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name+"_csv", filename.split(".")[0]+".csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for item in data:
            flattened_item = {
                'question': repr(item['question']),
                'hash': item['hash'],
                'response': repr(item['response']),
                'model': item['model'],
                'grammar': item['evaluation']['grammar'],
                'coherence': item['evaluation']['coherence'],
                'specificity': item['evaluation']['specificity']
            }
            writer.writerow(flattened_item)

def evaluations_jsonl_to_csv(comparison_task_name, model_ids, question_hashes):
    for filename in os.listdir(os.path.join(GENERATION_EVALUATION_DIR, comparison_task_name)):
        if filename.split(".")[0] in model_ids:
            evaluation_jsonl_to_csv(comparison_task_name, filename, question_hashes)

def compare_recipes(model_ids):
    fieldnames = ["model_id", "grammar", "coherence", "specificity", "sum", "0-7", "7-8", "8-9", "9-10"]
    data = {model_id: {k:0 for k in fieldnames[1:]} for model_id in model_ids}
    for training_similarity in ["0-7", "7-8", "8-9", "9-10"]:
        evaluation_result_dict = evaluation_results("inter_comparison", model_ids, get_question_hashes(f"split-{training_similarity}.jsonl"))
        for model_id in model_ids:
            data[model_id][training_similarity] = [evaluation_result[1] for evaluation_result in evaluation_result_dict["sum"] if evaluation_result[0]==model_id][0]
    evaluation_result_dict = evaluation_results("inter_comparison", model_ids, get_question_hashes(f"all.jsonl"))
    for model_id in model_ids:
        for k in ["grammar", "coherence", "specificity", "sum"]:
            data[model_id][k] = [evaluation_result[1] for evaluation_result in evaluation_result_dict[k] if evaluation_result[0]==model_id][0]

    with open(os.path.join(GENERATION_EVALUATION_DIR, "inter_comparison_csv", "recipes.csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
    
        for model_id in model_ids:
            flattened_item = {
                "model_id": model_id,
                **{k:round(data[model_id][k], 1) for k in fieldnames[1:]}
            }
            writer.writerow(flattened_item)

def compare_responses(model_ids):
    fieldnames = ["question"] + [f"{model_id} score" for model_id in model_ids] + model_ids
    for training_similarity in ["0-7", "7-8", "8-9", "9-10"]:
        data = get_data("inter_comparison", model_ids, get_question_hashes(f"split-{training_similarity}.jsonl"))
        top_models_for_questions = get_top_models_for_questions(data, "sum")
        with open(os.path.join(GENERATION_EVALUATION_DIR, "inter_comparison_csv", f"{training_similarity}-responses.csv"), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)      
            writer.writeheader()

            question_rows = []
        
            for question in top_models_for_questions.keys():
                flattened_item = {
                    "question": question,
                    **{model_id:repr(top_models_for_questions[question][model_id][1]) for model_id in model_ids},
                    **{f"{model_id} score":top_models_for_questions[question][model_id][0] for model_id in model_ids}
                }
                question_rows.append(flattened_item)

            question_rows.sort(key=lambda row: row[f"{model_ids[0]} score"], reverse=True)  # Sort in descending order

            for sorted_item in question_rows:
                writer.writerow(sorted_item)

if __name__ == "__main__":
    inner_comparison()
    inter_comparison()
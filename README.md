# TinyHelen
Code for paper [TinyHelen's First Curriculum: Training and Evaluating Tiny Language Models in a Simpler Language Environment](https://arxiv.org/abs/2501.00522).

☄️ We train and evaluate tiny language models (LMs) using a novel text dataset with systematically simplified vocabularies and linguistic structures, mimicking how children learn language through simplified environments as part of their initial curriculum.

☄️ **Motivation**

Training LMs and their application agents has become costly due to the large datasets and models required. Simplified language environments serve as a basic training and testing ground for LMs, presenting commonsense knowledge and communication skills in a plainer way. This can enhance learning efficiency, potentially reducing the model size and data volume needed for effective training. Strategies developed in these environments for smaller models, agents, and datasets may be applicable to larger, more complex language environments.

☄️ **What Has Been Done**

🔘 We introduce a "no noise, low complexity" principle, supporting our proposed text simplification pipeline that transforms the original training data of LMs into a version with reduced noise and complexity, which have been proven to enhance the learning efficiency of LMs.

🔘 We implement this pipeline to create a leaner dataset suite, the first to retain the traditional LM training dataset composition and evaluation benchmark intentions while being significantly linguistically simpler. It consists of a 71M dataset for pre-training, a 7M dataset for instruction-tuning, a benchmark that evaluates general linguistic proficiency, and a benchmark for measuring instruction-following ability.

🔘 Despite their infancy and limited initial performance, our instruction-following models mark one of the initial steps toward developing purely text-based self-evolving agents, an initiative we name TinyHelen.

🔘 The leaner datasets offer a testing ground for low-resource scenarios, enabling us to preliminarily evaluate how model architectures and curriculum learning strategies at the pre-training stage affect language modeling performance. Experiments show that transformer decoder llama preforms better on leaner-glue than state space model decoder mamba, controlling the model size (14M parameters) and pre-training data (100M tokens scale). Moreover, we find iteratively introducing training instances based on the LM's perplexity can reduce both pre-training steps and data requirements.

## Experiments
### Environment Setup
```bash
conda create -n leaner python=3.11.5.  
conda activate leaner  
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  
pip install -r requirements.txt  
python -m spacy download en_core_web_sm  

import nltk
nltk.download('punkt')

git clone https://github.com/state-spaces/mamba.git
cd ~/mamba
pip install .
```
### Exp1: Comparing Model Architectures with the Leaner Dataset Suite
We will use llama as an example. Experiments for other models can be implemented likewise.
```bash
bash scripts/pre-train/llama/run_clm_llama.sh
bash scripts/pre-train/llama/run_clm_llama_ori.sh
bash scripts/glue-eval/llama/run_glue_from_scratch_llama.sh
bash scripts/glue-eval/llama/run_glue_llama.sh
bash scripts/glue-eval/llama/run_glue_llama_ori.sh
```
### Exp2: Evaluating the Instruction-following Ability of Tiny LMs Trained with Different Data Recipes
```bash
bash scripts/pre-train/llama/run_clm_llama.sh
bash scripts/pre-train/llama/run_clm_llama_ori.sh
bash scripts/instruction-tune/run_clm_instruction.sh
python pipeline.py
python compare_instruction_following_generation.py
```
### Exp3: Investigate Curriculum Learning Strategies for Pre-training with Tiny Proxy Models
```bash
bash scripts/curriculum-learning/iterative_full_repeated.sh
bash scripts/curriculum-learning/iterative_full_random.sh
bash scripts/curriculum-learning/iterative.sh
bash scripts/glue-eval/curriculum-learning/iterative-no-training.sh
bash scripts/glue-eval/curriculum-learning/iterative.sh
```
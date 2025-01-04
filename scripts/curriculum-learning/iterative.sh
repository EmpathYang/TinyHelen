export CUDA_VISIBLE_DEVICES=0
INIT_STRATEGIES=("random" "average_sentence_tokens" "loss" "lm_loss")
UPDATE_STRATEGIES=("random" "average_sentence_tokens" "loss" "lm_loss")
RUN_NAMES=("iter-random" "iter-sentlen" "iter-selfloss" "iter-lmloss")
for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME=${RUN_NAMES[i]}
    INIT_STRATEGY=${INIT_STRATEGIES[i]}
    UPDATE_STRATEGY=${UPDATE_STRATEGIES[i]}
    python run_clm_curriculum_learning.py \
        --debugging false \
        --report_to wandb \
        --config_name config/llama/llama-2K-1M.json \
        --tokenizer_name tokenizer/gpt-2K \
        --data_dir data/leaner \
        --set_splits web book wiki textbook conversation \
        --corpus_total_token_num_choices 100M \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 512 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --eval_accumulation_steps 1 \
        --evaluation_strategy steps \
        --eval_steps 20 \
        --logging_steps 20 \
        --save_steps 100 \
        --max_steps 2500 \
        --warmup_steps 100 \
        --run_name  $RUN_NAME \
        --learning_rate 1e-2 \
        --seed  65 \
        --cache_dir /your/cache/dir \
        --output_dir /your/output/dir \
        --curriculum_data_dir /your/curriculum/output/data/dir \
        --dataset_update_strategy iterative \
        --iterative_debug false \
        --iterative_stopping_patience 1 \
        --iterative_stopping_threshold 0 \
        --iterative_init_strategy $INIT_STRATEGY \
        --iterative_update_strategy $UPDATE_STRATEGY \
        --iterative_reference_data_dir data/curriculum-learning \
        --metric_for_best_model loss \
        --reverse_difficulty false \
        --wandb_project_name leaner-curriculum-iterative
done
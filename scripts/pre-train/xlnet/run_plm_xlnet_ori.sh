export CUDA_VISIBLE_DEVICES=0,1,2,3
LEARNING_RATES=("5e-6" "5e-5" "5e-4" "1e-3" "5e-3" "5e-2")
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    python run_plm.py \
        --debugging false \
        --report_to wandb \
        --config_name config/xlnet/xlnet-2K-10M.json \
        --tokenizer_name tokenizer/xlnet-2K \
        --data_dir data/ori \
        --set_splits web book wiki textbook conversation \
        --corpus_total_token_num_choices 100M \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --eval_accumulation_steps 1 \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --logging_steps 100 \
        --save_steps 500 \
        --max_steps 10000 \
        --warmup_steps 100 \
        --run_name  10M-xlnet-ori-100M-natural-language-2K-"$LEARNING_RATE" \
        --learning_rate $LEARNING_RATE \
        --seed  65 \
        --cache_dir /your/cache/dir \
        --output_dir /your/output/dir \
        --wandb_project_name leaner-xlnet
done
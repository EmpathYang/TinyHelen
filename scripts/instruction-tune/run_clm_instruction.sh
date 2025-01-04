export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LEARNING_RATES=("5e-6" "5e-5" "5e-4" "1e-3" "5e-3" "5e-2")
DATA_DIRS=("data/leaner" "data/ori")
PRETRAINED_MODELS=("/your/leaner/model/dir" "/your/ori/model/dir")
MODEL_NAMES=("llama-leaner-leaner" "llama-ori-leaner" "llama-leaner-ori" "llama-ori-ori")
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for i in "${!DATA_DIRS[@]}"; do
        for j in "${!PRETRAINED_MODELS[@]}"; do
            DATA_DIR=${DATA_DIRS[i]}
            PRETRAINED_MODEL=${PRETRAINED_MODELS[j]}
            MODEL_NAME=${MODEL_NAMES[i*2+j]}
            python run_clm.py \
                --debugging false \
                --report_to wandb \
                --model_name_or_path $PRETRAINED_MODEL \
                --data_dir $DATA_DIR \
                --set_splits instruction \
                --do_train \
                --do_eval \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 4 \
                --gradient_accumulation_steps 8 \
                --eval_accumulation_steps 1 \
                --evaluation_strategy steps \
                --eval_steps 20 \
                --logging_steps 20 \
                --save_steps 100 \
                --max_steps 1000 \
                --warmup_steps 10 \
                --run_name  "$MODEL_NAME"-"$LEARNING_RATE" \
                --learning_rate $LEARNING_RATE \
                --seed  65 \
                --cache_dir /your/cache/dir \
                --output_dir /your/output/dir \
                --wandb_project_name leaner-instruct
        done
    done
done
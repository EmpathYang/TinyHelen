export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME=bert-1e-3-10000
TASKS=("cola" "sst2" "mrpc" "qqp" "stsb" "mnli" "qnli" "rte")
EVAL_STEPS=(30 250 10 1250 20 1250 350 10)
NUM_TRAIN_EPOCHS=(30 3 30 3 30 3 3 50)
LEARNING_RATES=("5e-6" "1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3")
RANDOM_SEEDS=(1 53 65 256 9264)
METRICS_FOR_BEST_MODEL=("matthews_correlation" "accuracy" "combined_score" "combined_score" "combined_score" "accuracy" "accuracy" "accuracy")
for i in "${!TASKS[@]}"; do
  for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
    TASK_NAME=${TASKS[i]}
    EVAL_STEP=${EVAL_STEPS[i]}
    NUM_TRAIN_EPOCH=${NUM_TRAIN_EPOCHS[i]}
    METRIC_FOR_BEST_MODEL=${METRICS_FOR_BEST_MODEL[i]}
    python run_glue.py \
      --seed $RANDOM_SEED \
      --model_name_or_path /your/model/dir \
      --task_name "$TASK_NAME" \
      --train_file data/leaner/glue/"$TASK_NAME"_train.jsonl \
      --validation_file data/leaner/glue/"$TASK_NAME"_validation.jsonl \
      --test_file data/leaner/glue/"$TASK_NAME"_test.jsonl \
      --do_train \
      --do_eval \
      --do_predict \
      --evaluation_strategy steps \
      --eval_steps $EVAL_STEP \
      --logging_steps $EVAL_STEP \
      --save_steps $EVAL_STEP \
      --max_seq_length 256 \
      --per_device_train_batch_size 256 \
      --gradient_accumulation_steps 1 \
      --learning_rate $LEARNING_RATE \
      --num_train_epochs $NUM_TRAIN_EPOCH \
      --run_name "$MODEL_NAME"-"$TASK_NAME"-"$LEARNING_RATE"-"$RANDOM_SEED" \
      --output_dir /your/output/dir/leaner/glue/"$MODEL_NAME"/leaner/"$TASK_NAME"/"$LEARNING_RATE"/"$RANDOM_SEED" \
      --save_total_limit 3 \
      --load_best_model_at_end \
      --metric_for_best_model $METRIC_FOR_BEST_MODEL \
      --cache_dir /your/cache/dir \
      --wandb_project_name leaner-glue-bert-"$TASK_NAME"
    done
  done
done
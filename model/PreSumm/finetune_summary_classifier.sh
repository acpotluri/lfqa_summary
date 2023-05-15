# finetune PreSumm extractive model

MODEL_PATH=""
PRETRAINED_MODEL_PATH=""
LOG_FILE_PATH=""

python src/train.py -task ext \
-mode train -bert_data_path \
bert_data/summary_finetune \
-ext_dropout 0.1 \
-model_path $MODEL_PATH \
-train_from $PRETRAINED_MODEL_PATH \
-lr 2e-3 \
-visible_gpus 2,3 \
-report_every 50 \
-save_checkpoint_steps 5000 \
-batch_size 16 \
-train_steps 50000 \
-accum_count 2 \
-log_file $LOG_FILE_PATH \
-use_interval true \
-warmup_steps 1000 \
-max_pos 512

# run finetuned PreSumm extractive model

MODEL_PATH="checkpoints/model_step_50000.pt"

LOG_FILE_PATH="logs"

# val result
python src/train.py -task ext \
-mode validate -bert_data_path \
bert_data/summary_finetune \
-ext_dropout 0.1 \
-model_file $MODEL_PATH \
-batch_size 16 \
-log_file $LOG_FILE_PATH \
-csv_output_file presumm_finetuned_val_predictions.csv

# test result
python src/train.py -task ext \
-mode test -bert_data_path \
bert_data/summary_finetune \
-ext_dropout 0.1 \
-test_from $MODEL_PATH \
-batch_size 16 \
-log_file $LOG_FILE_PATH \
-csv_output_file presum_finetuned_test_predictions.csv


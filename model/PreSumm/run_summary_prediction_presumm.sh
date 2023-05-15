# run PreSumm extractive model

MODEL_PATH="checkpoints/bertext_cnndm_transformer.pt"

LOG_FILE_PATH="logs"

# val result
python3 src/train.py -task ext \
-mode validate -bert_data_path \
bert_data/summary_finetune \
-ext_dropout 0.1 \
-model_file $MODEL_PATH \
-batch_size 16 \
-log_file $LOG_FILE_PATH \
-csv_output_file presumm_val_predictions.csv

# test result
python3 src/train.py -task ext \
-mode test -bert_data_path \
bert_data/summary_finetune \
-ext_dropout 0.1 \
-test_from $MODEL_PATH \
-batch_size 16 \
-log_file $LOG_FILE_PATH \
-csv_output_file presumm_test_predictions.csv


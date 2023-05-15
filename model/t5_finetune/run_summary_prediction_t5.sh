# run t5 role model on summary val / test data

# set directories
VAL_OUTPUT_DIR="results/val_result"
TEST_OUTPUT_DIR="results/test_result"
MODEL_PATH="checkpoints"

python run_hf_seq_to_seq.py \
--train_file data/train_summary_data.csv \
--validation_file data/validation_summary_data.csv \
--test_file data/validation_summary_data.csv \
--output_dir $VAL_OUTPUT_DIR \
--do_predict \
--overwrite_output_dir \
--evaluation_strategy epoch \
--predict_with_generate \
--num_train_epoch 0 \
--model_name_or_path $MODEL_PATH

python run_hf_seq_to_seq.py \
--train_file data/train_summary_data.csv \
--validation_file data/validation_summary_data.csv \
--test_file data/test_summary_data.csv \
--output_dir $TEST_OUTPUT_DIR \
--do_predict \
--overwrite_output_dir \
--evaluation_strategy epoch \
--predict_with_generate \
--num_train_epoch 0 \
--model_name_or_path $MODEL_PATH
# T5 large for summary prediction on summary dataset
# set output dir here
OUTPUT_DIR="results"

python run_hf_seq_to_seq.py \
--train_file data/train_summary_data.csv \
--validation_file data/validation_summary_data.csv \
--test_file data/test_summary_data.csv \
--model_name_or_path t5-large \
--output_dir $OUTPUT_DIR \
--do_train \
--do_eval \
--overwrite_output_dir \
--evaluation_strategy epoch \
--predict_with_generate \
--num_train_epoch 30 \
--learning_rate 1e-4 \
--save_strategy epoch \
--logging_strategy epoch \
--load_best_model_at_end \
--metric_for_best_model eval_macro_f1 \
--save_total_limit 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 # effective batch size 16
# lfqa_summary

## Introduction
This is the repository for annotated data and model for this paper: </br>

> Abhilash Potluri, Fangyuan Xu and Eunsol Choi. Concise Answers to Complex Questions: Summarization of Long-Form Answers. In: Proceedings of ACL. 2023.
> 
## Data

All our annotated data is stored in the `data` folder, split into a train/dev/test split where each example is a json with the following fields:
* `type`: The type of the annotation, all data should have `summary` as the value.
* `dataset`: The dataset this QA pair belongs to, one of [`NQ`, `ELI5`, `Web-GPT`].
* `q_id`: The question id, same as the original NQ or ELI5 dataset.
* `a_id`: The answer id, same as the original ELI5 dataset. For NQ, we populate a dummy `a_id` (1).
* `question`: The question.
* `answer_paragraph`: The answer paragraph.
* `answer_sentences`: The list of answer sentences, tokenzied from the answer paragraph.
* `summary_sentences`: The list of summary sentence index (starting from 1).
* `is_summary_count`: The list of count of annotators selecting this sentence as summary for the sentence in `answer_sentences`.
* `is_summary_1`: List of boolean value indicating whether annotator one selected the corresponding sentence as a summary sentence.
* `is_summary_2`: List of boolean value indicating whether annotator two selected the corresponding sentence as a summary sentence.
* `is_summary_3`: List of boolean value indicating whether annotator three selected the corresponding sentence as a summary sentence.

The dataset can also be found on [HuggingFace](https://huggingface.co/datasets/abhilashpotluri/lfqa_summary).

## Model
Each model sits in a different directory with its own environment file. Install conda and run the following command:

```bash
conda env create -f environment.yml
```

### T5

`cd models/t5_finetune/`

**Finetuning**

`./train_t5_summary_prediction.sh`

**Inference**

Download the model from [link](https://drive.google.com/file/d/1NtI2Xr9N5MO42VEbUl13XAX1NVKhwCqT/view?usp=sharing) and put it under the `t5_finetune/` folder.

`./run_summary_prediction_t5.sh`

### PreSumm

Download the models from [link](https://drive.google.com/file/d/1u2_roU53mjhtBInVnIV6tMl_aYSNqmZw/view?usp=sharing) and put it under the `PreSumm/` folder.

`cd models/PreSumm/`

**Finetuning**

`./finetune_summary_classifier.sh`

**Inference**

For the PreSumm model, run 

`./run_summary_prediction_presumm.sh`

For the finetuned model, run 

`./run_summary_prediction_presumm_finetuned.sh`

After getting the prediction results, run below script to generated the thresholded predicitons. 

```
python select_threshold_and_generate_test_pred.py 
--val_pred_path <path to val results> \
--test_pred_path <path to test results> \
--output_val_file_path <path to write thresholded val results> \
--output_test_file_path <path to write thresholded test results>
```

### Pegasus

To run the zero-shot Pegasus model run the following command from the `model/Pegasus` (where model name is the checkpoint from [HuggingFace](https://huggingface.co/models?search=pegasus) you would like to use):

```
python pegasus.py --batch_size <chosen batch size> --model_name <chosen checkpoint name>
```

### GPT-3

To run zero-shot with GPT-3 (specifically text-davinci-002), you need to have an OpenAI account and save your API key as the environment variable `OPENAI_API_KEY`. Then navigate to the `model/GPT` folder and then you can just run:

```
python gpt_zeroshot.py
```

## Automatic Evaluation

Each model has its own automatic evaluation script in the `autoEval` folder to compute the ROUGE scores of the predicted summaries (among other statistics like average length).

For example, to evaluate the T5 model output, you would run ```python t5.py``` and then it will also create a formatted csv that you can use as the input to bertScore.py in order to compute the BERTScore of the T5 model outputs.

## Human Evaluation

We also provide the html template (in `humanEval/study_template.html`) of the annotation interface which we used for the human study.

All the collected annotations are provided in `humanEval/results.csv` (worker IDs are hashed to preserve anonymity).

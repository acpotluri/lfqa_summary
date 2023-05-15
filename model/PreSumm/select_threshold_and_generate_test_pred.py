""" Generate test prediction after selecting threshold based on validation set"""
import ast
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
import utils
import tqdm


def generate_prediction_with_threshold(pred_path, threshold, output_file_path):
    pred_df = pd.read_csv(pred_path)
    pred_df['summary_sent_idx'] = pred_df['sent_idx'].map(lambda data: [int(idx) for idx in data.split(',')])
    q_level_pred = []
    for _, row in pred_df.iterrows():
        scores = row['sent_scores'].split(",")
        q_level_pred.append({
            'q_id': row['q_id'],
            'a_id': row['a_id'],
            'answer_summaries': row['answer_summaries'],
            'sent_idx': ','.join([str(i + 1) for (i, score) in enumerate(scores) if float(score) >= threshold])
        })
    # output
    pd.DataFrame.from_records(q_level_pred).to_csv(output_file_path, index=False)


def process_raw_data(data_df):
    data_df['is_summary_count'] = data_df['is_summary_count'].map(lambda data: data)
    data_df['summary_sentence_1'] = data_df['is_summary_count'].map(lambda data: [i+1 for i, vote in enumerate(data)
                                                                                                  if int(vote) > 0])
    data_df['summary_sentence_2'] = data_df['is_summary_count'].map(lambda data: [i+1 for i, vote in enumerate(data)
                                                                                                  if int(vote) > 1])
    data_df['summary_sentence_3'] = data_df['is_summary_count'].map(lambda data: [i+1 for i, vote in enumerate(data)
                                                                                                  if int(vote) > 2])
    data_df['num_answer_sentence'] = data_df['answer_sentences'].map(lambda data:len(data))
    return data_df

def main():
    argparse = ArgumentParser()
    argparse.add_argument("--val_pred_path", dest='val_pred_path',required=True)
    argparse.add_argument("--test_pred_path", dest='test_pred_path',required=True)
    argparse.add_argument("--output_val_file_path", dest='output_val_file_path', required=True)
    argparse.add_argument("--output_test_file_path", dest='output_test_file_path', required=True)
    argparse.add_argument("--with_question", dest='with_question', action='store_true', default=False)
    args = argparse.parse_args()

    summary_valid_df = pd.read_json('../../data/dev.json').transpose().reset_index(drop=True)
    pred_valid = pd.read_csv(args.val_pred_path)
    pred_valid['summary_sent_idx'] = pred_valid['sent_idx'].map(lambda data: [int(idx) for idx in data.split(',')])
    pred_valid['q_id'] = pred_valid['q_id'].astype(str)
    pred_valid['a_id'] = pred_valid['a_id'].astype(str)
    summary_valid_df['q_id'] = summary_valid_df['q_id'].astype(str)
    summary_valid_df['a_id'] = summary_valid_df['a_id'].astype(str)
    joined_with_labels = pred_valid.join(summary_valid_df.set_index(['q_id', 'a_id']), on=['q_id', 'a_id'],
                                         how='left')

    # flatten
    sentence_level_pred_with_labels = []
    for _, row in joined_with_labels.iterrows():
        scores = row['sent_scores'].split(",")
        votes = row['is_summary_count']
        # print(row, votes)
        for i, vote in enumerate(votes):
            vote = int(vote)
            sentence_level_pred_with_labels.append({
                'pred': float(scores[i]),
                'vote': vote if vote > 0 else 3,
                'label': 1 if vote > 0 else 0
            })
    sentence_level_pred_with_labels_df = pd.DataFrame.from_records(sentence_level_pred_with_labels)
    # choose threshold
    thresholds = np.array(sentence_level_pred_with_labels_df.sort_values('pred')['pred'])
    # select ten percentiles
    # thresholds = [thres[0] for thres in np.array_split(thresholds, 10)]

    res = []
    summary_valid_df = process_raw_data(summary_valid_df)
    for t in tqdm.tqdm(thresholds):
        # print(t)
        # create pred label

        sentence_level_pred_with_labels_df_copy = utils.compile_dataframe_from_presumm_prediction(args.val_pred_path, with_question=args.with_question) # sentence_level_pred_with_labels_df.copy()
        # print(sentence_level_pred_with_labels_df_copy)
        sentence_level_pred_with_labels_df_copy['summary_sent_idx'] = sentence_level_pred_with_labels_df_copy.apply(lambda data:
                                                                                                                    [i+1 for (i, score) in enumerate(data['sent_scores'].split(","))
                                                                                                                     if float(score) >= t], axis=1)
        # p, r, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        #     y_pred=sentence_level_pred_with_labels_df_copy['pred_label'],
        #     y_true=sentence_level_pred_with_labels_df_copy['label'],
        #     # sample_weight=sentence_level_pred_with_labels_df_copy['vote'],
        #     average='binary')

        p, r, f1, num_question, num_summary_sentences, _ = utils.evaluate_summary_selection_take_max(sentence_level_pred_with_labels_df_copy,
                                                                          summary_valid_df)
        res.append({'threshold': t,
                    'p': p,
                    'r': r,
                    'f1': f1})

    # take the one that gives max f1 on validation
    thres = pd.DataFrame.from_records(res).sort_values('f1').reset_index(drop=True).iloc[-1]['threshold']
    print(pd.DataFrame.from_records(res).sort_values('f1'))
    # generate val thresholded prediction
    generate_prediction_with_threshold(args.val_pred_path, thres, args.output_val_file_path)

    # generate test thresholded prediction
    generate_prediction_with_threshold(args.test_pred_path, thres, args.output_test_file_path)



if __name__ == "__main__":
    main()

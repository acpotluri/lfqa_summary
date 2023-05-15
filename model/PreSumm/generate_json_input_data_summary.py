""" Generate input data for summary finetuning"""
import ast
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
import pandas as pd
import json
from datetime import date
import os

def main():
    argparse = ArgumentParser()
    argparse.add_argument("--input_path", dest='input_path',
                          default='raw_data')
    argparse.add_argument("--output_path", dest='output_path',
                          default='json_data')

    argparse.add_argument("--add_question", dest='add_question', action='store_true')
    args = argparse.parse_args()

    for split in ['train', 'validation', 'test']:
        print(" >> processing {} data ".format(split))
        detailed_data_df = pd.read_csv('{}/{}_summary.csv'.format(args.input_path, split))
        detailed_data_df['tokenized_question'] = detailed_data_df['question'].map(lambda data: word_tokenize(data))
        # construct output
        output_list = []
        for _, row in detailed_data_df.iterrows():
            answer_sents = ast.literal_eval(row['answer_sentences'])
            counts = ast.literal_eval(row['is_summary_count'])
            srcs, labels = [], []
            for i, answer in enumerate(answer_sents):
                src = word_tokenize(answer)
                # train with summary selected by any annotator
                if int(counts[i]) > 0:
                    labels.append(i) # this index starts from 0
                srcs.append(src)
            if args.add_question:
                # prepend question
                srcs = [row['tokenized_question']] + src
                # need to shift one
                # TODO
                # labels = [i+1 for i in labels]
            data = {
                'q_id': str(row['q_id']),
                'a_id': str(row['a_id']),
                'src': srcs,
                # dummy tgt
                'tgt': srcs,
                # convert this to id
                'labels': labels
            }
            output_list.append(data)

        # write output
        output_file = '{}/summary_finetune.{}.0.json'.format(args.output_path, split)
        if args.add_question:
            output_file = '{}/summary_finetune_question_prepended.{}.0.json'.format(args.output_path, split)
        with open(output_file, 'w') as fout:
            json.dump(output_list, fout)

        print(">> output {} data to {}".format(len(output_list), output_file))

if __name__ == "__main__":
    main()
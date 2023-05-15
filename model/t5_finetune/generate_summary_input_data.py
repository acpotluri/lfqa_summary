"""Generate input data for T5 model on summary prediction"""

from argparse import ArgumentParser
import pandas as pd
from datetime import date
import os
import ast

def main():
    argparse = ArgumentParser()
    # input file is one of the summary file
    argparse.add_argument("--input_file", dest='input_file', required=True)
    # output file is in format of T5 input
    argparse.add_argument("--output_file", dest='output_file', required=True)
    args = argparse.parse_args()

    # first read the summary file
    question_level_summary_data = pd.read_csv(args.input_file)

    res_data = []
    for _, row in question_level_summary_data.iterrows():
        input_line = [row['question']]
        is_summary_count = ast.literal_eval(row['is_summary_count'])
        target_line = []
        target_line_all_labels = []
        answer_sentences = ast.literal_eval(row['answer_sentences'])
        for idx, sentence in enumerate(answer_sentences):
            sep_token = '[{}]'.format(idx)
            input_line.append(sep_token)
            input_line.append(sentence)
            target_line.append(sep_token)
            if int(is_summary_count[idx]) > 0:
                target_line.append('Answer (Summary)')
            else:
                target_line.append('Others') # dummy
            target_line_all_labels.append(sep_token)

        input_line = ' '.join(input_line)
        target_line = ' '.join(target_line)

        res_data.append({
            # first column is input
            'input_txt': input_line,
            # second column is target
            'target_txt': target_line,
            # meta info
            # 'q_id': row['q_id'],
            # 'a_id': row['a_id'],
        })
    # write output
    pd.DataFrame.from_records(res_data).to_csv(args.output_file, index=False)
    print(">> output to {}".format(args.output_file))

if __name__ == "__main__":
    main()




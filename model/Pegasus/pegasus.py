from typing import List
import torch
from torch.utils.data import DataLoader
from rouge import Rouge
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import argparse
from tqdm import tqdm
import csv
import random
from datasets import load_dataset
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt

def _parse_args():
    """
    Command-line arguments to the system
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size; 1 by default and you do not need to batch unless you want to')
    parser.add_argument('--model_name', type=str, default="google/pegasus-cnn_dailymail")
    args = parser.parse_args()
    return args

args = _parse_args()
torch.cuda.empty_cache()

from typing import List

class LongFormQAExample:
    """
    Data wrapper for a single example of long form qa summary.
    Attributes:
        question (string): words for the question
        answer (string): words in the initial long form answer
        summary (string): random annotater summary of the answer
    """

    def __init__(self, qid, question, answer, summary, ansSentences):
        self.qid = qid
        self.question = question
        self.answer = answer
        self.summary = summary
        self.ansSentences = ansSentences

    def __repr__(self):
        return repr(self.qid) + "; question=" + repr(self.question) + "; answer=" + repr(self.answer) + "; summary=" + repr(self.summary)

    def __str__(self):
        return self.__repr__()

class LongFormQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, data):
        self.encodings = encodings
        self.summaries = [ex.summary for ex in data]
        self.qids = [ex.qid for ex in data]
        self.answers = [ex.answer for ex in data]
        self.questions = [ex.question for ex in data]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.summaries[idx]
        item['qids'] = self.qids[idx]
        item['answers'] = self.answers[idx]
        item['questions'] = self.questions[idx]
        return item

    def __len__(self):
        return len(self.summaries)

def read_examples(infile: str) -> List[LongFormQAExample]:
    """
    :param infile: file to read from
    :return: a list of LongFormQAExample parsed from the file
    """
    import json, random
    with open(infile, 'r') as f:
        exs = []
        allData = json.load(f)
        for data_id in allData.keys():
            ex = allData[data_id]
            q = ex['question']
            ans = ex['answer_sentences']
            rand_annotater = random.choice(['is_summary_1', 'is_summary_2', 'is_summary_3'])
            label = ' '.join([ans[i] if x else '' for i, x in enumerate(ex[rand_annotater])])
            exs.append(LongFormQAExample(data_id, q, ' '.join(ans), label, ans))
        f.close()
        return exs

data = read_examples('test.json')

model_name = args.model_name
tokenizer = PegasusTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
rouge_calc = Rouge()

data_ids = set()

def zero_shot_evaluate(encodings, tot_data, fname, firstCall=False, containsQuestion=True):
    num_grams = 2
    encoded_dataset = LongFormQADataset(encodings, tot_data)
    data_loader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=True)
    csv_data = []
    uni_bigram_scores = []
    compression_ratios = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            translated = model.generate(batch["input_ids"])
            pred_summary = tokenizer.batch_decode(translated, skip_special_tokens=True)
            for i in range(len(pred_summary)):
                model_pred = pred_summary[i]
                model_input = batch["answers"][i]
                if containsQuestion:
                    model_pred = model_pred.replace(batch["questions"][i], '')
                    model_input = batch["questions"][i] + " " + batch["answers"][i]
                model_pred = model_pred.replace("<n>", '')
                row = {}
                row['Data ID'] = batch["qids"][i]
                row["Question"] = batch["questions"][i]
                row["Random Annotater Summary"] = batch["labels"][i]
                row["Pegasus Summary"] = model_pred
                row["Answer"] = batch["answers"][i]

                def strip_punctuation(s):
                    newS = s.replace(' .', '')
                    newS = newS.replace(' ,', '')
                    newS = newS.replace('.', '')
                    newS = newS.replace(',', '')
                    return newS
                ans_bigrams = set(Counter(ngrams(strip_punctuation(model_input).split(), num_grams)).keys())
                pred_bigrams = set(Counter(ngrams(strip_punctuation(model_pred).split(), num_grams)).keys())
                compression_ratios.append(len(model_pred.split(' ')) / len(batch["answers"][i].split(' ')))
                cnt = 0
                for k in pred_bigrams:
                    if k not in ans_bigrams:
                        cnt += 1
                if len(pred_bigrams) > 0:
                    row["Unique Bigram"] = cnt / len(pred_bigrams)
                    rscores = rouge_calc.get_scores(model_pred, batch["labels"][i])[0]
                    row["Rouge-L Recall"] = rscores['rouge-l']['r']
                    row["Rouge-L Precision"] = rscores['rouge-l']['p']
                    row["Rouge-L F1"] = rscores['rouge-l']['f']
                    row["Rouge-1 Recall"] = rscores['rouge-1']['r']
                    row["Rouge-1 Precision"] = rscores['rouge-1']['p']
                    row["Rouge-1 F1"] = rscores['rouge-1']['f']
                    row["Rouge-2 Recall"] = rscores['rouge-2']['r']
                    row["Rouge-2 Precision"] = rscores['rouge-2']['p']
                    row["Rouge-2 F1"] = rscores['rouge-2']['f']
                else:
                    row["Unique Bigram"] = 0.0
                    row["Rouge-L Recall"] = 0.0
                    row["Rouge-L Precision"] = 0.0
                    row["Rouge-L F1"] = 0.0
                    row["Rouge-1 Recall"] = 0.0
                    row["Rouge-1 Precision"] = 0.0
                    row["Rouge-1 F1"] = 0.0
                    row["Rouge-2 Recall"] = 0.0
                    row["Rouge-2 Precision"] = 0.0
                    row["Rouge-2 F1"] = 0.0
                uni_bigram_scores.append(row["Unique Bigram"])
                csv_data.append(row)
    
    with open(fname, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(csv_data[0].keys()))
        writer.writeheader()
        # to guarantee the same questions are printed for both input type
        if firstCall:
            rows = random.sample(csv_data, min(len(csv_data), 2000))
            writer.writerows(rows)
            for row in rows:
                data_ids.add(row['Data ID'])
        else:
            rows = []
            for row in csv_data:
                if row['Data ID'] in data_ids:
                    rows.append(row)
            writer.writerows(rows)

    print(rouge_calc.get_scores([ex["Pegasus Summary"] if len(ex["Pegasus Summary"]) > 1 else "N/A" for ex in csv_data], [ex["Random Annotater Summary"] for ex in csv_data], avg=True))
    avg_len = 0
    for ex in csv_data:
        avg_len += len(ex["Pegasus Summary"].split(" "))
    avg_len /= len(csv_data)
    print("Average Length of Summary: " + str(avg_len))
    print("Average Unique Bigram Score: " + str(sum(uni_bigram_scores)/len(uni_bigram_scores)))
    print("Average Compression Ratio: " + str(sum(compression_ratios)/len(compression_ratios)))
    if containsQuestion:
        plt.title("CNN Input = Question + Answer Compression, avg = " + str(sum(compression_ratios) / len(compression_ratios)))
    else:
        plt.title("CNN Input = Answer Compression, avg = " + str(sum(compression_ratios) / len(compression_ratios)))
    plt.xlabel('Token Level Compression')
    plt.ylabel('Number of Examples')
    plt.xlim(0, 1)
    plt.hist(compression_ratios, bins=20)
    if containsQuestion:
        plt.savefig('qa.png')
    else:
        plt.savefig('a.png')
    plt.clf()



encodings = tokenizer([ex.answer for ex in data], truncation=True, padding="longest", return_tensors="pt").to(device)
zero_shot_evaluate(encodings, data, "no_question_summaries.csv", True, False)

random.shuffle(data)
encodings = tokenizer([ex.question + " " + ex.answer for ex in data], truncation=True, padding="longest", return_tensors="pt").to(device)
zero_shot_evaluate(encodings, data, "with_question_summaries.csv", True)

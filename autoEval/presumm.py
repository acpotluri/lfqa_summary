import csv, re, json
import numpy as np
from rouge import Rouge
from sklearn.metrics import f1_score, precision_score, recall_score

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

rouge = Rouge()
idData = {}
qData = {}
with open("test.json", 'r') as f:
    exs = []
    idData = json.load(f)
    for k in idData:
        qData[idData[k]["question"]] = idData[k]

options = ["is_summary_1", "is_summary_2", "is_summary_3"]

data = {}
posLabels = {}
with open('presumm_test.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ansSent = row['summary_sentences'][1:-1].split(",")
        if len(ansSent[0]) > 0:
            summaries = [int(x) - 1 for x in ansSent]
            ans = row['answer_paragraph'].strip(' ')
            ans = re.split(r'\n|\.', ans)
            idx = row['q_id']
            s = ''
            for x in summaries:
                s += ans[x].lower()
            data[idx] = (summaries, ans, s)
            labs = []
            ex = qData[row["question"]]
            ansSent = ex["answer_sentences"]
            for opt in options:
                labs.append(' '.join([ansSent[i] if ex[opt][i] else "" for i in range(len(ex[opt]))]))
            posLabels[idx] = labs


def calc(fname, oname):
    print(fname)
    with open(fname, 'r') as predf:
        r = csv.DictReader(predf)
        preds = []
        labels = []
        lengths = []
        slens = []
        acc = []
        precisions = []
        recalls = []
        exact_cnt = 0
        for row in r:
            idx = row['q_id']
            pred = row['answer_summaries'].replace('<q>', ' ')
            predIdx = []
            if len(row['sent_idx'].split(",")[0]) > 0:
                predIdx = [int(x) - 1 for x in row['sent_idx'].split(",")]
            if idx in data and len(pred) > 0:
                l = data[idx][2]
                sumIdx = data[idx][0]
                ansLen = len(data[idx][1])
                predBool = [1 if i in predIdx else 0 for i in range(ansLen)]
                sumBool = [1 if i in sumIdx else 0 for i in range(ansLen)]
                acc.append(f1_score(sumBool, predBool))
                p1 = precision_score(sumBool, predBool)
                r1 = recall_score(sumBool, predBool)
                if p1 == 1.0 and r1 == 1.0:
                    exact_cnt += 1
                precisions.append(p1)
                recalls.append(r1)
                if len(l) > 0:
                    posLabs = posLabels[idx]
                    f1s = np.array([rouge.get_scores(pred, posLabs[i])[0]["rouge-l"]['f'] for i in range(3)])
                    preds.append(pred)
                    labels.append(posLabs[np.argmax(f1s)])
                    # labels.append(l)
            lengths.append(len(pred.split(' ')))
            slens.append(len(pred.split('.')) - 1)
        print("pred", rouge.get_scores(preds, labels, avg=True))
        print("pred avg len: ", sum(lengths) / len(lengths))
        print("pred avg sentence len: ", sum(slens) / len(slens))
        print("recall: ", sum(recalls) / len(recalls))
        print("precision: ", sum(precisions) / len(precisions))
        print("f1: ", sum(acc) / len(acc))
        print("exact match", exact_cnt / len(preds))

    with open(oname, 'w') as csvw:
        rows = [{'label': labels[i], 'summary': preds[i]} for i in range(len(labels))]
        writer = csv.DictWriter(csvw, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


calc('pred_presumm.csv', 'no_finetune_presumm_parsed.csv')
calc('finetune_presumm.csv', 'finetune_presumm_parsed.csv')

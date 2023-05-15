import csv, json
import numpy as np
from rouge import Rouge

rouge = Rouge()

posLabels = {}
options = ["is_summary_1", "is_summary_2", "is_summary_3"]

with open("test.json", 'r') as f:
    exs = []
    idData = json.load(f)
    for k in idData:
        ex = idData[k]
        labs = []
        ansSent = ex["answer_sentences"]
        for opt in options:
            labs.append(' '.join([ansSent[i] if ex[opt][i] else "" for i in range(len(ex[opt]))]))
        posLabels[k] = labs

def calc(fname, oname):
    print(fname)
    preds = []
    labels = []
    lens = []
    slens = []
    with open(fname, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            id = row["Data ID"]
            p = row['Pegasus Summary']
            l = row['Random Annotater Summary']
            if len(p) > 0 and len(l) > 0:
                posLabs = posLabels[id]
                f1s = np.array([rouge.get_scores(p, posLabs[i])[0]["rouge-l"]['f'] for i in range(3)])
                preds.append(p)
                labels.append(posLabs[np.argmax(f1s)])
            lens.append(len(p.split(' ')))
            slens.append(len(p.split('.')) - 1)
    print("pred", rouge.get_scores(preds, labels, avg=True))
    print("pred len", sum(lens) / len(lens))
    print("pred sent len", sum(slens) / len(slens))

    with open(oname, 'w') as csvw:
        rows = [{'label': labels[i], 'summary': preds[i]} for i in range(len(labels))]
        writer = csv.DictWriter(csvw, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



calc('no_question_summaries.csv', 'cnn_no_question_parsed.csv')
calc('with_question_summaries.csv', 'cnn_question_parsed.csv')

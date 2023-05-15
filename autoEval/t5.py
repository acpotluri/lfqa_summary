import csv, re, json
import numpy as np
from rouge import Rouge

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
	
        inp_format = ex["question"]
        for i in range(len(ansSent)):
            inp_format += " [" + str(i) + "] " + ansSent[i]
        posLabels[inp_format] = labs

options = ["is_summary_1", "is_summary_2", "is_summary_3"]

preds = []
with open('generated_predictions.txt', 'r') as f:
    for row in f:
        preds.append(row.strip('\n'))

def extract(s):
    c = '\['
    inds = [i.start() for i in re.finditer(c, s)]
    l = []
    n = len(inds)
    for i in range(n):
        ind1 = inds[i] + 3
        ind2 = inds[i + 1] if i != n - 1 else len(s)
        l.append(s[ind1:ind2].strip(' '))
    return l

ansTok = 'Answer (Summary)' 
with open('test_summary_data.csv', 'r') as csvfile:
    r = csv.DictReader(csvfile)
    labels = []
    summaries = []
    lengths = []
    labs = []
    accuracy = [] 
    slengths = []
    for i, row in enumerate(r):
        inp = str(row['input_txt'])
        label = str(row['target_txt'])
        summary = preds[i]

        ans = extract(inp)
        label = extract(label)
        summary = extract(summary)
        
        l = ''
        s = ''
        lab = [0 for _ in range(len(label))]
        acc = [0 for _ in range(len(label))]
        for i in range(len(label)):
            if label[i] == ansTok:
                l += ' ' + ans[i]
                lab[i] = 1
            if summary[i] == ansTok:
                s += ' ' + ans[i]
                acc[i] = 1
        labs.append(lab)
        accuracy.append(acc)
        if len(s) > 0 and len(l) > 0:
            posLabs = posLabels[inp]
            f1s = np.array([rouge.get_scores(s, posLabs[i])[0]["rouge-l"]['f'] for i in range(3)])
            labels.append(posLabs[np.argmax(f1s)])
            summaries.append(s)
        lengths.append(len(s.split(' ')))
        slengths.append(len(s.split('.')) - 1)
    
    with open('t5_parsed.csv', 'w') as csvw:
        rows = [{'label': labels[i], 'summary': summaries[i]} for i in range(len(labels))]
        writer = csv.DictWriter(csvw, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
 
    print("pred", rouge.get_scores(summaries, labels, avg=True))
    print("pred avg len:", sum(lengths) / len(lengths))
    print("pred sentence avg len:", sum(slengths) / len(slengths))
    
    from sklearn.metrics import f1_score, recall_score, precision_score
    def calc(vals):
        f1s = []
        recalls = []
        precisions = []
        exact_cnt = 0
        for i in range(len(labs)):
            p1 = precision_score(labs[i], vals[i])
            r1 = recall_score(labs[i], vals[i])
            if p1 == r1 and p1 == 1.0:
                exact_cnt += 1
            precisions.append(p1)
            recalls.append(r1)
            f1s.append(f1_score(labs[i], vals[i]))
        print("classification precision:", sum(precisions) / len(precisions))
        print("classification recall:", sum(recalls) / len(recalls))
        print("classification f1:", sum(f1s) / len(f1s))
        print("exact match:", exact_cnt / len(vals))

    calc(accuracy)

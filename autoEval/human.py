import csv, json, random
from rouge import Rouge
from sklearn.metrics import f1_score, precision_score, recall_score

rouge = Rouge()

options = ["is_summary_1", "is_summary_2", "is_summary_3"]

with open("test.json", 'r') as f:
    labels = []
    preds = []
    lengths = []
    slengths = []
    recalls = []
    precisions = []
    f1s = []
    exactCnt = 0
    allData = json.load(f)
    for dataid in allData:
        ex = allData[dataid]
        ans = ex["answer_sentences"]
        label_idx = random.choice(options)
        lab = ' '.join([ans[i] if ex[label_idx][i] else "" for i in range(len(ex[label_idx]))])
        labones = [1 if ex[label_idx][i] else 0 for i in range(len(ex[label_idx]))]
        def compute(pred_idx):
            global exactCnt
            pred = ' '.join([ans[i] if ex[pred_idx][i] else "" for i in range(len(ex[pred_idx]))])
            predones = [1 if ex[pred_idx][i] else 0 for i in range(len(ex[pred_idx]))]
            labels.append(lab)
            preds.append(pred)
            lengths.append(len(pred.split(' ')))
            slengths.append(len(pred.split('.')) - 1)
            f1s.append(f1_score(labones, predones))
            p1 = precision_score(labones, predones)
            r1 = recall_score(labones, predones)
            if p1 == 1.0 and r1 == 1.0:
                exactCnt += 1
            precisions.append(p1)
            recalls.append(r1)

        
        for opt in options:
            if label_idx != opt:
                compute(opt)
    
    print("pred", rouge.get_scores(preds, labels, avg=True))
    print("pred len", sum(lengths) / len(lengths))
    print("pred sent len", sum(slengths) / len(slengths))
    print("recall: ", sum(recalls) / len(recalls))
    print("precision: ", sum(precisions) / len(precisions))
    print("f1: ", sum(f1s) / len(f1s))
    print("exact match", exactCnt / len(preds))


    with open("human_parsed.csv", 'w') as csvw:
        rows = [{'label': labels[i], 'summary': preds[i]} for i in range(len(labels))]
        writer = csv.DictWriter(csvw, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

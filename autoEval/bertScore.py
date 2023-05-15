from datasets import load_metric
import csv

def get_avg(l):
	return sum(l) / len(l)

def calc(fname):
	print(fname)
	metric = load_metric("bertscore")
	with open(fname, 'r') as f:
		r = csv.DictReader(f)
		for row in r:
			metric.add(row['summary'], row['label'])
	results = metric.compute(model_type="bert-base-uncased")
	print("precision ", get_avg(results['precision']))
	print("recall ", get_avg(results['recall']))
	print("f1 ", get_avg(results['f1']))
	print()

calc("gpt_limit_parsed.csv")
calc("gpt_no_limit_parsed.csv")

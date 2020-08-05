
import json
import csv
from utils import parse_statement


with open('../dataset/train.json') as f:
    dataset = f.readlines()
train_dataset = [json.loads(data) for data in dataset]
with open('../dataset/train.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['label', 'body'])
    for i, data in enumerate(train_dataset):
        x, y = parse_statement(data['code']), data['nl']
        try:
            writer.writerow([x, y])
        except:
            writer.writerow([data['code'], data['nl']])

with open('../dataset/test.json') as f:
    dataset = f.readlines()
test_dataset = [json.loads(data) for data in dataset]
with open('../dataset/test.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['label', 'body'])
    for i, data in enumerate(test_dataset):
        x, y = parse_statement(data['code']), data['nl']
        try:
            writer.writerow([x, y])
        except:
            writer.writerow([data['code'], data['nl']])

print('perpeare the data successful')
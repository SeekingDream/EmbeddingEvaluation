
import json
import csv

with open('../dataset/train.json') as f:
    dataset = f.readlines()
train_dataset = [json.loads(data) for data in dataset]
with open('../dataset/train.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    for i, data in enumerate(train_dataset):
        x, y = data['code'], data['nl']
        writer.writerow([x, y])


with open('../dataset/test.json') as f:
    dataset = f.readlines()
test_dataset = [json.loads(data) for data in dataset]
with open('../dataset/test.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    for i, data in enumerate(test_dataset):
        x, y = data['code'], data['nl']
        writer.writerow([x, y])

print('perpeare the data successful')
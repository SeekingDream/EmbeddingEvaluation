import pickle
import matplotlib.pyplot as plt

from vector_evaluation.experiment import Metric

metric_key =  [
    'acc',
    'map',
    'ndcg',
    'mrr',
    'recall_3',
    'prec_3'
]

with open('exp.res', 'rb') as f:
    ori_result = pickle.load(f)

result = sorted(ori_result, key=lambda student: student[2])


for k in metric_key:
    y_1 = []
    y_2 = []
    for r in result:
        y_1.append(r[0].res[k])
        y_2.append(r[0].random[k])
    plt.plot(y_1, 'r', label='app')
    plt.plot(y_2, 'b', label='random')
    plt.title(k)
    plt.show()

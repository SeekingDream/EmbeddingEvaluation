import os
import torch
from vector_evaluation import SemanticCosine
import joblib

vec_dir = '../code-authorship/Pytorch-RNN-text-classification/gen/'
dict_list, vec_list = {}, {}

for file_name in os.listdir(vec_dir):
    if '.pkl' in file_name:
        dict_val = joblib.load(vec_dir + file_name)
        key_val = file_name.replace('.pkl', '')
        dict_list[key_val] = dict_val
    else:
        vec_val = torch.load(vec_dir + file_name)
        vec_list[file_name] = vec_val.detach().cpu().numpy()

vec = []
for k in vec_list:
    vec.append(vec_list[k])

metric = SemanticCosine(vec, sampling_num= 1000)
metric.calculate_score()
metric = SemanticCosine(vec, sampling_num= 5000)
metric.calculate_score()
metric = SemanticCosine(vec, sampling_num= 10000)
metric.calculate_score()
import pickle
import json
import os
from multiprocessing import Process


def build_dict(dataset):
    token2index, path2index, func2index = {}, {}, {}
    for i, data in enumerate(dataset):
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        if func_name not in func2index:
            func2index[func_name] = len(func2index)
        for context in code_context:
            st, path, ed = context.split(',')
            if st not in token2index:
                token2index[st] = len(token2index)
            if ed not in token2index:
                token2index[ed] = len(token2index)
            if path not in path2index:
                path2index[path] = len(path2index)
    with open('dataset/java-small-preprocess/tk.pkl', 'wb') as f:
        pickle.dump([token2index, path2index, func2index], f)
    print("finish dictionary build")


def tk2index(tk_dict, k):
    if k not in tk_dict:
        return len(tk_dict)
    return tk_dict[k]


def norm_data(data_type):
    file_name = 'dataset/java-small/java-small.' + data_type + '.c2v'
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    with open('dataset/java-small-preprocess/tk.pkl', 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
    newdataset = []
    for i, data in enumerate(dataset):
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        label = tk2index(func2index, func_name)
        newdata = []
        for context in code_context:
            st, path, ed = context.split(',')
            newdata.append(
                [tk2index(token2index, st), tk2index(path2index, path), tk2index(token2index, ed)]
            )
        newdataset.append([newdata, label])
    save_file = 'dataset/java-small-preprocess/' + data_type + '.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(newdataset, f)
    print("finish normalize dataset", data_type)


def main():
    with open('dataset/java-small/java-small.train.c2v', 'r') as f:
        dataset = f.readlines()
        print('data number is ', len(dataset))
    if not os.path.isdir('dataset/java-small-preprocess'):
        os.mkdir('dataset/java-small-preprocess')
    build_dict(dataset)
    norm_data('train')
    norm_data('val')
    norm_data('test')


if __name__ == '__main__':
    main()

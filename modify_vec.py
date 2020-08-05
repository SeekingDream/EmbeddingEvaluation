import torch
import numpy as np


def modify_vec(file_name):
    word2index, vec = torch.load(file_name)
    if '____UNKNOW____' in word2index:
        return
    for k in word2index:
        if type(word2index[k]) is int:
            word2index[k] += 2
        else:
            word2index[k] = word2index[k].index + 2
    word2index['____UNKNOW____'] = 0
    word2index['____PAD____'] = 1
    dim = vec.shape[1]
    tmp_vec = np.random.rand(2, dim)
    vec = np.concatenate([tmp_vec, vec])
    torch.save([word2index, vec], file_name)
    print('successful', file_name)


def main():
    file_name = './embedding_vec/code_vec/doc2vec.vec'
    modify_vec(file_name)


if __name__ == '__main__':
    main()

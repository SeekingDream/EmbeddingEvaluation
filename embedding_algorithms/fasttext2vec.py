import logging
import fasttext
import pandas as pd
import codecs
from embedding_algorithms import BasicEmbedding
import numpy as np
from gensim.models import word2vec
import os

class FastEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(FastEmbedding, self).__init__(file_name, dataset, vocab, vec_dim, epoch)

    def generate_embedding(self):
        file_list = word2vec.PathLineSentences(self.file_name).input_files
        res = []
        for file_name in file_list:
            with open(file_name, 'r') as f:
                res.append(f.read())
        if not os.path.isdir('../tmp_res'):
            os.mkdir('../tmp_res')
        with open('../tmp_res/tmp_file', 'w') as f:
            f.writelines(res)
        classifier = fasttext.train_unsupervised(
            input='../tmp_res/tmp_file',
            dim=self.vec_dim,
            epoch=self.epoch,
            minCount=10,
            thread=10
        )
        return self.get_res(classifier)

    def get_res(self, classifier):
        words = classifier.words
        vec = np.random.rand(len(words) + 2, self.vec_dim)
        word2index = {
            '____UNKNOW____': 0,
            '____PAD____': 1
        }
        for i, word in enumerate(words):
            word2index[word] = i + 2
            vec[i + 2] = classifier.get_word_vector(word)
        return word2index, vec

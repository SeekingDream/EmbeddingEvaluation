import logging
import fasttext
import pandas as pd
import codecs
from embedding_algorithms import BasicEmbedding
import numpy as np

class FastEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(FastEmbedding, self).__init__(dataset, vocab, vec_dim, epoch)
        self.train_file = file_name

    def generate_embedding(self):
        classifier = fasttext.train_unsupervised(
            input=self.train_file, minCount=1, dim=self.vec_dim, epoch=self.epoch)
        return self.get_res(classifier)

    def get_res(self, classifier):
        words = classifier.words
        vec = np.zeros([len(words), self.vec_dim])
        word2index = {}
        for i, word in enumerate(words):
            word2index[word] = i
            vec[i] = classifier.get_word_vector(word)
        return word2index, vec

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn


class BasicEmbedding(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(BasicEmbedding, self).__init__()
        self.dataset = dataset
        self.vocab = vocab
        self.vec_dim = vec_dim
        self.epoch = epoch
        self.file_name = file_name

    @abstractmethod
    def generate_embedding(self, model_type):
        return None, None

    def save_embedding(self, file_name):
        torch.save([self.vec], file_name)


class TokenClass:
    def __init__(self, dataset):
        self.word2index = {
            '____UNKNOW____': 0,
            '____PAD____':    1,
        }
        for data in dataset:
            for tk in data:
                if tk not in self.word2index:
                    self.word2index[tk] = len(self.word2index)

    def __getitem__(self, ky):
        if ky not in self.word2index:
            return 0
        else:
            return self.word2index[ky]

from gensim.models import word2vec
from embedding_algorithms import BasicEmbedding
import numpy as np


def trans_vocab(vocab, vectors):
    new_vocab = {
        '____UNKNOW____': 0,
        '____PAD____': 1
    }
    for tk in vocab:
        new_vocab[tk] = vocab[tk].index + 2
    dim = vectors.shape[1]
    tmp_vec = np.random.rand(2, dim)
    vec = np.concatenate([tmp_vec, vectors])
    return new_vocab, vec


class Word2VecEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(Word2VecEmbedding, self).__init__(file_name, dataset, vocab, vec_dim, epoch)

    def generate_embedding(self):
        sentences = word2vec.PathLineSentences(self.file_name)
        model = word2vec.Word2Vec(
            sentences=sentences,
            size=self.vec_dim,
            min_count=10,
            workers=10
        )
        model.train(
            sentences=sentences,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=self.epoch
        )
        return trans_vocab(model.wv.vocab, model.wv.vectors)

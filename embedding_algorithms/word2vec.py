from gensim.models import word2vec
from embedding_algorithms import BasicEmbedding
import numpy as np
from utils import trans_vocab


class Word2VecEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(Word2VecEmbedding, self).__init__(file_name, dataset, vocab, vec_dim, epoch)

    def generate_embedding(self, model_type):
        sentences = word2vec.PathLineSentences(self.file_name)
        model = word2vec.Word2Vec(
            sentences=sentences,
            size=self.vec_dim,
            sg=model_type,
            min_count=1,
            workers=10
        )
        model.train(
            sentences=sentences,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=self.epoch
        )
        return trans_vocab(model.wv.vocab, model.wv.vectors)

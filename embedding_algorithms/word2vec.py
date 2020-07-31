from gensim.models import word2vec
from embedding_algorithms import BasicEmbedding


class Word2VecEmbedding(BasicEmbedding):
    def __init__(self, dataset, vocab, vec_dim, epoch):
        super(Word2VecEmbedding, self).__init__(dataset, vocab, vec_dim, epoch)

    def generate_embedding(self):
        model = word2vec.Word2Vec(self.dataset, size=self.vec_dim, min_count=1)
        model.train(self.dataset, total_examples=model.corpus_count, epochs=self.epoch)
        return self.trans_vocab(model.wv.vocab), model.wv.vectors

    def trans_vocab(self, vocab):
        res = {}
        for tk in vocab:
            res[tk] = vocab[tk]
        return res

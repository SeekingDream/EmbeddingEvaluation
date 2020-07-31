from gensim.models import doc2vec
from embedding_algorithms import BasicEmbedding


class Doc2VecEmbedding(BasicEmbedding):
    def __init__(self, dataset, vocab, vec_dim, epoch):
        super(Doc2VecEmbedding, self).__init__(dataset, vocab, vec_dim, epoch)

    def generate_embedding(self):
        sentences = [doc2vec.TaggedDocument(sentence, str(index)) for index, sentence in enumerate(self.dataset)]
        model = doc2vec.Doc2Vec(sentences, size=self.vec_dim, min_count=1)
        model.train(sentences, total_examples=model.corpus_count, epochs=self.epoch)
        return self.trans_vocab(model.wv.vocab), model.wv.vectors

    def trans_vocab(self, vocab):
        res = {}
        for tk in vocab:
            res[tk] = vocab[tk]
        return res

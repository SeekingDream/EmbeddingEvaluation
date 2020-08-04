from gensim.models import doc2vec
from embedding_algorithms import BasicEmbedding


def trans_vocab(vocab):
    res = {}
    for tk in vocab:
        res[tk] = vocab[tk]
    return res


class Doc2VecEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(Doc2VecEmbedding, self).__init__(file_name, dataset, vocab, vec_dim, epoch)

    def generate_embedding(self):
        #sentences = [doc2vec.TaggedDocument(sentence, str(index)) for index, sentence in enumerate(self.dataset)]
        model = doc2vec.Doc2Vec(
            corpus_file=self.file_name,
            size=self.vec_dim,
            min_count=10,
            workers=10
        )
        model.train(
            corpus_file=self.file_name,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=self.epoch
        )
        return trans_vocab(model.wv.vocab), model.wv.vectors

from gensim.models import doc2vec, word2vec
from gensim.models.doc2vec import TaggedDocument
from embedding_algorithms import BasicEmbedding
import numpy as np
import gensim
import os

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


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        LabeledSentence = gensim.models.doc2vec.LabeledSentence
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(
                words=doc.split(), tags=self.labels_list[idx]
            )


class Doc2VecEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(Doc2VecEmbedding, self).__init__(file_name, dataset, vocab, vec_dim, epoch)

    def generate_embedding(self):
        sentences = word2vec.PathLineSentences(self.file_name)
        docLabels = sentences.input_files
        data = []
        for doc in docLabels:
            with open(doc) as f:
                doc_data = f.read()
                data.append(doc_data)
        it = LabeledLineSentence(data, docLabels)
        model = doc2vec.Doc2Vec(
            documents=it,
            vector_size=self.vec_dim,
            min_count=10,
            workers=10
        )
        model.train(
            documents=it,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=self.epoch
        )
        return trans_vocab(model.wv.vocab, model.wv.vectors)

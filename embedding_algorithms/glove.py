import numpy as np
from gensim.models import word2vec
from embedding_algorithms import BasicEmbedding
from glove import Corpus, Glove 


def trans_vocab(vocab, vectors):
    new_vocab = {
        '____UNKNOW____': 0,
        '____PAD____': 1
    }
    for tk in vocab:
        new_vocab[tk] = vocab[tk] + 2 # add the special tokens
    dim = vectors.shape[1]
    tmp_vec = np.random.rand(2, dim)
    vec = np.concatenate([tmp_vec, vectors])
    return new_vocab, vec


class GloVeEmbedding(BasicEmbedding):
    def __init__(
        self, file_name, dataset, vocab, vec_dim, learning_rate=0.05, 
        window=10, epoch=1, no_threads=4, verbose=True
    ):
        super(GloVeEmbedding, self).__init__(
            file_name, dataset, vocab, vec_dim, 
            learning_rate, window, epoch, no_threads, verbose
        )

    def generate_embedding(self):
        sentences = word2vec.PathLineSentences(self.file_name)

        # Training the corpus to generate the co-occurance matrix which is used in GloVe
        corpus = Corpus() # Creating a corpus object
        corpus.fit(sentences, window=self.window) 

        # Training GloVe model
        glove = Glove(
            no_components=self.vec_dim, 
            learning_rate=self.learning_rate
        ) 
        glove.fit(
            corpus.matrix, epochs=self.epoch, 
            no_threads=self.no_threads, verbose=self.verbose
        )
        glove.add_dictionary(corpus.dictionary)

        return trans_vocab(glove.dictionary, glove.word_vectors)








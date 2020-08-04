from embedding_algorithms import *
import torch
import os


def train_vec(vec_dim, epoch, data_file):
    dir_name = '../embedding_vec/' + str(vec_dim) + '_' + str(epoch) + '/'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)



    m = Word2VecEmbedding(data_file, None, None, vec_dim=vec_dim, epoch=epoch)
    vocab, vec = m.generate_embedding()
    torch.save([vocab, vec], dir_name + 'word2vec.vec')
    print('get the pre-trained vectors from', m.__class__.__name__, 'vector dim is %d, epoch is %d' % (vec_dim, epoch))

    m = Doc2VecEmbedding(data_file, None, None, vec_dim=vec_dim, epoch=epoch)
    vocab, vec = m.generate_embedding()
    torch.save([vocab, vec], dir_name + 'doc2vec.vec')
    print('get the pre-trained vectors from', m.__class__.__name__, 'vector dim is %d, epoch is %d' % (vec_dim, epoch))

    m = FastEmbedding(data_file, None, None, vec_dim=vec_dim, epoch=epoch)
    vocab, vec = m.generate_embedding()
    torch.save([vocab, vec], dir_name + 'fasttext.vec')
    print('get the pre-trained vectors from', m.__class__.__name__, 'vector dim is %d, epoch is %d' % (vec_dim, epoch))


def main():
    if not os.path.isdir('../embedding_vec'):
        os.mkdir('../embedding_vec')
    data_file = '../dataset/nlp_train_embedding.data'
    for dim in [100, 200, 300]:
        for epoch in [1]:
            train_vec(dim, epoch, data_file)


if __name__ == '__main__':
    main()

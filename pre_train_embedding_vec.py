from embedding_algorithms import *
import torch
import os
import gensim

EMBED_DIR = './embedding_vec'
DATA_DIR = './dataset/code_embedding'

EMBEDDING_LIST = [
    #Word2VecEmbedding,
    #Doc2VecEmbedding,
    #FastEmbedding,
]


def train_vec(vec_dim, epoch, data_dir):
    dir_name = EMBED_DIR + str(vec_dim) + '_' + str(epoch) + '/'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    m = Word2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    vocab, vec = m.generate_embedding()
    torch.save([vocab, vec], dir_name + m.__class__.__name__ + '.vec')
    print(
        'get the pre-trained vectors from', 
        m.__class__.__name__, 
        'vector dim is %d, epoch is %d' % (vec_dim, epoch)
    )


def main():
    # if not os.path.isdir(EMBED_DIR):
    #     os.mkdir(EMBED_DIR)

    for dim in [100, 200, 300]:
        for epoch in [1]:
            train_vec(dim, epoch, DATA_DIR)


if __name__ == '__main__':
    main()

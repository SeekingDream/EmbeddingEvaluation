from embedding_algorithms import *
import torch
import os

EMBED_DIR = 'embedding_vec'
DATA_DIR = 'dataset/code_embedding_java_small'


EMBEDDING_LIST = [
    Word2VecEmbedding,
    Doc2VecEmbedding,
    FastEmbedding,
    GloVeEmbedding,
]


def produce_vec(model, model_type, dir_name):
    vocab, vec = model.generate_embedding(model_type=model_type)
    if type(model_type) is str:
        save_name = dir_name + model.__class__.__name__ + model_type + '.vec'
    else:
        save_name = dir_name + model.__class__.__name__ + str(model_type) + '.vec'
    torch.save([vocab, vec], save_name)
    print(save_name)


def train_vec(vec_dim, epoch, data_dir, dir_name):
    model = Word2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    produce_vec(model, model_type=0, dir_name=dir_name)

    model = Word2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    produce_vec(model, model_type=1, dir_name=dir_name)

    model = Doc2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    produce_vec(model, model_type=0, dir_name=dir_name)

    model = Doc2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    produce_vec(model, model_type=1, dir_name=dir_name)

    model = FastEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    produce_vec(model, model_type='skipgram', dir_name=dir_name)
    produce_vec(model, model_type='cbow', dir_name=dir_name)

    model = GloVeEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
    produce_vec(model, model_type='None', dir_name=dir_name)


def main():
    for dim in [100,  200,  300]:
        for epoch in [1]:
            dir_name = EMBED_DIR + str(dim) + '_' + str(epoch) + '/'
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            train_vec(dim, epoch, DATA_DIR, dir_name)


if __name__ == '__main__':
    main()

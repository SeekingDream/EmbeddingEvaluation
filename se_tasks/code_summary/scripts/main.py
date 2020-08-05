from se_tasks.code_summary.scripts.CodeLoader import CodeLoader
from torch.utils.data import DataLoader
from se_tasks.code_summary.scripts.Code2VecModule import Code2Vec
import pickle
import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import datetime
import argparse
from utils import set_random_seed
import numpy as np
import os


def my_collate(batch):
    x, y = zip(*batch)
    sts, paths, eds = [], [], []
    for data in x:
        st, path, ed = zip(*data)
        sts.append(torch.tensor(st, dtype=torch.int))
        paths.append(torch.tensor(path, dtype=torch.int))
        eds.append(torch.tensor(ed, dtype=torch.int))

    length = [len(i) for i in sts]
    sts = rnn_utils.pad_sequence(sts, batch_first=True).long()
    eds = rnn_utils.pad_sequence(eds, batch_first=True).long()
    paths = rnn_utils.pad_sequence(paths, batch_first=True).long()
    return (sts, paths, eds), y, length


def train_model(tk_path, train_path, test_path, embed_dim, embed_type, vec_path, experiment_name):
    with open(tk_path, 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
        embed = None
    if embed_type == 0:
        token2index, embed = torch.load(vec_path)
        tmp_vec = np.random.randn(2, embed_dim)
        embed = np.concatenate([embed, tmp_vec], axis=0)
        print('load existing embedding vectors, name is ', vec_path)
    elif embed_type == 1:
        print('create new embedding vectors, training from scratch')
    elif embed_type == 2:
        embed = torch.randn([len(token2index) + 2, args.embed_dim]).cuda()
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float).cuda()
        assert embed.size()[1] == embed_dim
    if not os.path.exists('../result'):
        os.mkdir('../result')

    nodes_dim, paths_dim, output_dim = len(token2index), len(path2index), len(func2index)

    model = Code2Vec(nodes_dim + 2, paths_dim + 2, embed_dim, output_dim + 1, embed)
    criterian = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dataset = CodeLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=my_collate)
    print('train data size is ', len(train_dataset))

    #test_dataset = CodeLoader(test_path)
    #test_loader = DataLoader(test_dataset, batch_size=1000, collate_fn=my_collate)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        acc = 0
        st_time = datetime.datetime.now()
        for i, ((sts, paths, eds), y, length) in enumerate(train_loader):
            sts = sts.to(device)
            paths = paths.to(device)
            eds = eds.to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            pred_y = model(sts, paths, eds, length, device)
            loss = criterian(pred_y, y)
            loss.backward()
            optimizer.step()
            pos, pred_y = torch.max(pred_y, 1)
            acc += torch.sum(pred_y == y)
        ed_time = datetime.datetime.now()
        print('epoch', epoch, 'acc:', acc.float().item() / len(train_dataset), 'cost time', ed_time - st_time)


def main(args_set):
    tk_path = args_set.tk_path
    train_path = args_set.train_data
    test_path = args_set.test_data
    embed_dim = args_set.embed_dim
    embed_type = args_set.embed_type
    vec_path = args_set.embed_path
    experiment_name = args.experiment_name
    train_model(tk_path, train_path, test_path, embed_dim, embed_type, vec_path, experiment_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch', default=16, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--classes', default=250, type=int, metavar='N', help='number of output classes')
    parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--embed_path', type=str, default='../../embedding_vec/100_1/fasttext.vec')
    parser.add_argument('--train_data', type=str, default='../data/java-small-preprocess/train.pkl')
    parser.add_argument('--test_data', type=str, default='../data/java-small-preprocess/val.pkl')
    parser.add_argument('--tk_path', type=str, default='../data/java-small-preprocess/tk.pkl')
    parser.add_argument('--embed_type', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code2vec')

    args = parser.parse_args()
    set_random_seed(10)
    main(args)
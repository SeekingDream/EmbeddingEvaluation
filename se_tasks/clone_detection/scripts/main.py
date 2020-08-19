import gc
import os
import argparse
import datetime
import numpy as np
import pickle
import torch
from torch import nn
import torch.backends.cudnn as cudnn
# import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_random_seed
from torch.utils.data import DataLoader


from se_tasks.clone_detection.scripts.model import CloneModel
from se_tasks.clone_detection.scripts.vocab import build_vocab, tokensize_dataset


def load_data(train_file, val_file, word2index, batch_size, max_num):
    def my_collate_fn(batch):
        return zip(*batch)
    train_data = tokensize_dataset(train_file, word2index, max_num)
    val_data = tokensize_dataset(val_file, word2index, max_num)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=my_collate_fn)
    print('load the dataset, train size', len(train_data),'test data size', len(val_data))
    return train_loader, val_loader


def perpare_exp_set(embed_type, ebed_path, train_path, embed_dim):
    if embed_type == 0:
        d_word_index, embed = torch.load(ebed_path)
        print('load existing embedding vectors, name is ', ebed_path)
    elif args.embedding_type == 1:
        d_word_index = build_vocab(train_path)
        embed = None
        print('create new embedding vectors, training from scratch')
    elif args.embedding_type == 2:
        d_word_index = build_vocab(train_path)
        embed = torch.randn([len(d_word_index), args.embedding_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')

    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float)
        assert embed.size(1) == embed_dim

    if not os.path.exists('se_tasks/clone_detection/result'):
        os.mkdir('se_tasks/clone_detection/result')
    return d_word_index, embed


def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    loss_list = []
    for i, data in tqdm(enumerate(train_loader)):
        node_1, graph_1, node_2, graph_2, label = data
        label = torch.tensor(label, dtype= torch.float, device=device)
        label = label * 2 - 1
        output = model(node_1, graph_1, node_2, graph_2, device)
        loss = criterion(output, label)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return loss_list


def test_model(val_loader, model, device):
    model.eval()
    acc = 0
    tp, fn, fp = 0, 0, 0
    for i, data in enumerate(val_loader):
        node_1, graph_1, node_2, graph_2, label = data
        label = torch.tensor(label, dtype=torch.float, device=device)
        output = model(node_1, graph_1, node_2, graph_2, device)
        output = (output > 0)
        acc += (output == label).sum().float()
        tp += ((output == 1) * (label == 1)).sum().float()
        fn += ((output == 0) * (label == 1)).sum().float()
        fp += ((output == 1) * (label == 0)).sum().float()
    acc = acc / len(val_loader.dataset)
    prec = tp / (tp + fn)
    recall = tp / (tp + fp)
    res = {'acc': acc.item(), 'p': prec.item(), 'r':recall.item()}
    return res


def main(arg_set):
    print("===> creating vocabs ...")
    train_path = arg_set.train_data
    val_path = arg_set.test_data
    pre_embedding_path = arg_set.embed_path
    embed_dim = arg_set.embed_dim

    d_word_index, embed = perpare_exp_set(
        arg_set.embed_type, pre_embedding_path, train_path, embed_dim
    )
    vocab_size = len(d_word_index)

    train_loader, val_loader = load_data(
        train_path, val_path, d_word_index, arg_set.batch, 20000
    )

    model = CloneModel(vocab_size, embed_dim, embedding_tensor=embed)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=arg_set.lr,
        weight_decay=arg_set.weight_decay
    )
    criterion = nn.MSELoss()
    # device = torch.device(int(arg_set.device))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    acc_curve = []
    for epoch in range(1, arg_set.epochs + 1):
        st = datetime.datetime.now()
        train_model(train_loader, model, criterion, optimizer, device)
        res = test_model(val_loader, model, device)
        ed = datetime.datetime.now()
        print(epoch, ' epoch cost time', ed - st, 'result is', res)
        acc_curve.append(res)

    save_name = 'se_tasks/clone_detection/result/' + arg_set.experiment_name + '.h5'
    res = {
        'word2index': d_word_index,
        'model': model.state_dict(),
        'acc_curve': acc_curve
    }
    torch.save(res, save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')

    parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--embed_path', type=str, default='embedding_vec/100_2/Word2VecEmbedding0.vec')
    parser.add_argument('--train_data', type=str, default='se_tasks/clone_detection/data/train.pkl')
    parser.add_argument('--test_data', type=str, default='se_tasks/clone_detection/data/test.pkl')
    parser.add_argument('--embed_type', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code2vec')
    parser.add_argument('--device', type=int, default=7)

    args = parser.parse_args()
    set_random_seed(10)
    main(args)
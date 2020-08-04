from __future__ import print_function

import gc
import os
import argparse
import datetime
import numpy as np
import joblib
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from gensim.models import word2vec

from vocab import VocabBuilder, GloveVocabBuilder
from dataloader import Word2vecLoader
from model import RNN
from util import AverageMeter, accuracy
from util import adjust_learning_rate

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2048, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=100, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=250, type=int, metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--glove', default='glove/glove.6B.100d.txt', help='path to glove txt')
parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--weight_name', type=str, default='1', help='model name')
parser.add_argument('--train_data', type=str, default='./train.tsv', help='model name')
parser.add_argument('--test_data', type=str, default='./test.tsv', help='model name')


args = parser.parse_args()

# create vocab
print("===> creating vocabs ...")
end = datetime.datetime.now()

v_builder, d_word_index, embed = None, None, None
train_path = args.train_data
test_path = args.test_data
dic_name = os.path.join('gen', args.weight_name+'.pkl')
weight_save_model = os.path.join('gen', args.weight_name)

try:
    d_word_index = joblib.load(dic_name)
    embed = torch.load(weight_save_model)
    print('load existing embedding vectors, name is ', args.weight_name)
except:
    v_builder = VocabBuilder(path_file=train_path)
    d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
    print('create new embedding vectors')

if not os.path.exists('gen'):
    os.mkdir('gen')
joblib.dump(d_word_index, dic_name, compress=3)

end = datetime.datetime.now()
train_loader = Word2vecLoader(train_path, d_word_index, batch_size=args.batch_size)
val_loader = Word2vecLoader(test_path, d_word_index, batch_size=args.batch_size)

vocab_size = len(d_word_index)


class Word2vecPredict(nn.Module):
    def __init__(self, d_word_index, token_vec):
        super(Word2vecPredict, self).__init__()
        vocab_size = len(d_word_index)
        if torch.is_tensor(token_vec):
            self.encoder = nn.Embedding(vocab_size, 100, padding_idx=0, _weight=token_vec)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, 100, padding_idx=0)

        self.linear = nn.Linear(100, len(d_word_index))

    def forward(self, x):
        vec = self.encoder(x)
        vec = torch.mean(vec, dim = 1)
        pred = self.linear(vec)
        return pred


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def test(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target, _) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))
    return top1.avg


model = Word2vecPredict(d_word_index, embed)
model = model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                             weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

print('training dataset size is ', train_loader.n_samples)


t1 = datetime.datetime.now()
for epoch in range(1, 10):
    st = datetime.datetime.now()
    train(train_loader, model, criterion, optimizer)
    res = test(val_loader, model, criterion)
    ed = datetime.datetime.now()
t2 = datetime.datetime.now()

weight_save_model = os.path.join('gen', args.weight_name)
torch.save(model.encoder.weight, weight_save_model)
print('result is ', res)
print('result is ', res, 'cost time', t2 - t1)
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

from vocab import  VocabBuilder, GloveVocabBuilder
from dataloader import TextClassDataLoader
from model import RNN
from util import AverageMeter, accuracy
from util import adjust_learning_rate



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=100, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=2, type=int, metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--glove', default='glove/glove.6B.100d.txt', help='path to glove txt')
parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--weight_name', type=str, default='d1_author_1', help='model name')
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
train_loader = TextClassDataLoader(train_path, d_word_index, batch_size=args.batch_size)
val_loader = TextClassDataLoader(test_path, d_word_index, batch_size=args.batch_size)


vocab_size = len(d_word_index)
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes, rnn_model=args.rnn,
            use_last=( not args.mean_seq),
            hidden_size=args.hidden_size, embedding_tensor=embed, num_layers=args.layers, batch_first=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss()

if args.cuda:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    criterion = criterion.cuda()


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target, seq_lengths) in enumerate(train_loader):
        # measure data loading time
        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input, seq_lengths)
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

        # measure elapsed time


def test(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target,seq_lengths) in enumerate(val_loader):
        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input,seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
    #print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


# training and testing
t1 = datetime.datetime.now()
for epoch in range(1, args.epochs+1):
    st = datetime.datetime.now()
    adjust_learning_rate(args.lr, optimizer, epoch)
    train(train_loader, model, criterion, optimizer)
    res = test(val_loader, model, criterion)
    ed = datetime.datetime.now()
t2 = datetime.datetime.now()

torch.save(model.encoder.weight, weight_save_model)
print('result is ', res, 'cost time', t2 - t1)

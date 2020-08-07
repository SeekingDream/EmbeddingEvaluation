# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill

from collections import OrderedDict

from tqdm import tqdm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors
from torchtext.data.metrics import bleu_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from se_tasks.comment_generate.scripts.options import train_opts
from se_tasks.comment_generate.scripts.options import model_opts
from se_tasks.comment_generate.scripts.model import Seq2seqAttn
from se_tasks.comment_generate.scripts.dataloader import CodeDataset


def id2w(pred, field):
    sentence = [field.vocab.itos[i] for i in pred]
    if '<eos>' in sentence:
        return sentence[:sentence.index('<eos>')]
    return sentence


class Trainer(object):
    def __init__(
            self, model, criterion, optimizer, scheduler, clip):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.n_updates = 0


    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, samples, tf_ratio, is_train=True):
        self.optimizer.zero_grad()
        if is_train:
            outs = self.model(samples.body, samples.label, tf_ratio)
        else:
            preds = self.model(samples.body, None, tf_ratio=0.0)
            preds = preds.max(2)[1].transpose(1, 0)
            return preds
        loss = self.criterion(
            outs.view(-1, outs.size(2)),
            samples.label.view(-1).cuda()
        )

        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.n_updates += 1
        return loss


def save_model(save_vars, filename):
    model_path = os.path.join(args.savedir, filename)
    torch.save(save_vars, model_path)


def save_vocab(savedir, fields):
    name, field = fields
    save_path = os.path.join(savedir, f"{name}_vocab.txt")
    with open(save_path, 'w') as fout:
        for w in field.vocab.itos:
            fout.write(w + '\n')


def save_field(savedir, fields):
    name, field = fields
    save_path = os.path.join(savedir, f"{name}.field")
    with open(save_path, 'wb') as fout:
        dill.dump(field, fout)


def main(args):
    #device = torch.device('cuda' if args.gpu else 'cpu')
    train_path, test_path = args.train_data, args.test_data

    # load data and construct vocabulary dictionary
    SRC = data.Field(
        init_token='<sos>',
        eos_token='<eos>',
        pad_token="____PAD____",
        unk_token="____UNKNOW____",
        lower=False
    )
    TGT = data.Field(
        init_token='<sos>',
        eos_token='<eos>',
        pad_token="____PAD____",
        unk_token="____UNKNOW____",
        lower=False
    )
    fields = [('src', SRC), ('tgt', TGT)]

    train_data = CodeDataset(train_path)
    train_data = data.Dataset(train_data, fields=[('body', SRC), ('label', TGT)])
    valid_data = CodeDataset(test_path)
    valid_data = data.Dataset(valid_data, fields=[('body', SRC), ('label', TGT)])
    print('training dataset size is', len(train_data))

    SRC.build_vocab(
        train_data.body, min_freq=args.src_min_freq,
        specials=['____UNKNOW____', '____PAD____'],
    )
    TGT.build_vocab(
        train_data.label, min_freq=args.tgt_min_freq,
        specials=['____UNKNOW____', '____PAD____'],
    )
    embed = None
    pre_embed_path = args.embed_path
    if args.embed_type == 0:
        d_word_index, embed = torch.load(pre_embed_path)
        SRC.vocab.set_vectors(d_word_index, embed, args.embedding_dim)
        print('load existing embedding vectors, name is ', pre_embed_path)
    elif args.embed_type == 1:
        print('create new embedding vectors, training from scratch')
    elif args.embed_type == 2:
        embed = torch.randn([len(SRC.vocab.stoi), args.embedding_dim]).cuda()
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float)
        assert embed.size()[1] == args.embedding_dim
    if not os.path.exists('../results'):
        os.mkdir('../results')
    SRC.vocab.UNK = SRC.unk_token
    TGT.vocab.UNK = TGT.unk_token
    SRC.vocab.stoi.default_factory = int
    TGT.vocab.stoi.default_factory = int
    print('source vob size is', len(SRC.vocab.stoi), 'target vob size is', len(TGT.vocab.stoi),)

    # set iterator
    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args.batch,
        sort_within_batch=True,
        sort_key=lambda x: len(x.body),
        repeat=False,
    )

    model = Seq2seqAttn(args, fields)
    model = model.cuda()
    print(model)
    print('')

    criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi['____PAD____']).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = nn.DataParallel(optimizer, device_ids=[1, 2, 3, 7])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    trainer = Trainer(model, criterion, optimizer, scheduler, args.clip)

    epoch = 1
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    best_loss = math.inf

    while epoch < max_epoch and trainer.n_updates < max_update \
            and args.min_lr < trainer.get_lr():

        # training
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            train_loss = 0.0
            trainer.model.train()
            for samples in pbar:
                # bsz = samples.src.size(1)
                loss = trainer.step(samples, args.tf_ratio)
                train_loss += loss.item()

                # setting of progressbar
                pbar.set_description(f"epoch {str(epoch).zfill(3)}")
                progress_state = OrderedDict(
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=len(samples),
                    lr=trainer.get_lr(),
                    clip=args.clip,
                    num_updates=trainer.n_updates)
                pbar.set_postfix(progress_state)
        train_loss /= len(train_iter)

        print(f"| epoch {str(epoch).zfill(3)} | train ", end="")
        print(f"| loss {train_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(train_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")

        # validation

        res_score = 0
        trainer.model.eval()
        for samples in valid_iter:
            preds = trainer.step(samples, tf_ratio=0.0, is_train=False)
            preds = [id2w(pred, TGT) for pred in preds]

            reference = samples.label.transpose(1, 0)
            reference = [id2w(pred, TGT) for pred in reference]
            score = bleu_score(preds, reference)
            res_score += score
        res_score = res_score / len(valid_iter)
        print(res_score)


        # saving model
        save_vars = {"train_args": args,
                     "state_dict": model.state_dict()}


        save_model(save_vars, "checkpoint_last.pt")

        # update
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    train_opts(parser)
    model_opts(parser)
    args = parser.parse_args()
    main(args)
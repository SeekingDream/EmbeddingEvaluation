# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill

from collections import OrderedDict

from tqdm import tqdm

# from torchtext import data
# from torchtext import vocab
# from torchtext import datasets
# from torchtext.vocab import Vectors
from torchtext.data.metrics import bleu_score

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from se_tasks.comment_generate.scripts.options import train_opts
from se_tasks.comment_generate.scripts.options import model_opts
from se_tasks.comment_generate.scripts.model import Seq2seqAttn
from se_tasks.comment_generate.scripts.vocab import VocabBuilder
from se_tasks.comment_generate.scripts.dataloader import TextClassDataLoader


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

    def step(self, src, tar, src_len, tar_len, tf_ratio):
        self.optimizer.zero_grad()
        outs = self.model(src, src_len, tar, tar_len, tf_ratio)
        loss = self.criterion(outs.view(-1, outs.size(2)), tar.view(-1))

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
    device = torch.device('cuda' if args.gpu else 'cpu')
    train_path, test_path = args.train_data, args.test_data

    v_builder = VocabBuilder(path_file=train_path)
    src_word_index, embed, tar_word_index, embed = v_builder.get_word_index()

    pre_embedding_path = args.embed_file
    if args.embed_type == 0:
        src_word_index, embed = torch.load(pre_embedding_path)
        print('load existing embedding vectors, name is ', pre_embedding_path)
    elif args.embed_type == 1:
        print('create new embedding vectors, training from scratch')
    elif args.embed_type == 2:
        embed = torch.randn([len(src_word_index), args.embedding_dim]).cuda()
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float).cuda()
        assert embed.size()[1] == args.embedding_dim

    if not os.path.exists('se_tasks/comment_generate/result'):
        os.mkdir('se_tasks/comment_generate/result')

    train_loader = TextClassDataLoader(train_path, src_word_index, tar_word_index, batch_size=args.batch)
    val_loader = TextClassDataLoader(test_path, src_word_index, tar_word_index, batch_size=args.batch)

    model = Seq2seqAttn(
        args,
        len(src_word_index),
        len(tar_word_index),
        device, embed
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    trainer = Trainer(model, criterion, optimizer, scheduler, args.clip)

    epoch = 1
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    best_loss = math.inf

    while epoch < max_epoch and trainer.n_updates < max_update \
            and args.min_lr < trainer.get_lr():
        # training
        trainer.model.train()
        train_loss = 0.0
        for (src, tar, src_len, tar_len) in train_loader:
            loss = trainer.step(src, tar, src_len, tar_len, args.tf_ratio)
            train_loss += loss.item()
        train_loss /= train_loader.batch_size

        # validation
        valid_loss = 0.0
        trainer.model.eval()
        for (src, tar, src_len, tar_len) in val_loader:
            loss = trainer.step(src, tar, src_len, tar_len, args.tf_ratio)
            valid_loss += loss.item()

        valid_loss /= val_loader.batch_size

        # print(f"| epoch {str(epoch).zfill(3)} | valid ", end="")
        # print(f"| loss {valid_loss:.{4}} ", end="")
        # print(f"| ppl {math.exp(valid_loss):.{4}} ", end="")
        # print(f"| lr {trainer.get_lr():.1e} ", end="")
        # print(f"| clip {args.clip} ", end="")
        # print(f"| num_updates {trainer.n_updates} |")
        #
        # # saving model
        # save_vars = {"train_args": args,
        #              "state_dict": model.state_dict()}
        #
        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     save_model(save_vars, 'checkpoint_best.pt')
        # save_model(save_vars, "checkpoint_last.pt")

        # update
        trainer.scheduler.step(valid_loss)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    train_opts(parser)
    model_opts(parser)
    args = parser.parse_args()
    main(args)


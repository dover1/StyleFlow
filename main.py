import os
import json

import argparse
import numpy as np
import torch
import random
import wandb

from train import train, pretrain
from data_loader import Vocab, get_dataloader
from model import TSST, Discriminator
from logger import Log
from config import LABEL_MAP, CONFIG_MAP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_dir(config):
    config.save_dir = os.path.join(config.save_dir, config.dataset,
                                   f'separate={config.separate}_flowType={config.flow_type}_flowNum={config.flow_nums}'
                                   f'_seed={config.seed}')
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    print('Save Dir:', config.save_dir)
    config.json_file = os.path.join(config.save_dir, 'config.json')

    ckpt_dir = os.path.join(config.save_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print('Ckpt Dir:', ckpt_dir)
    config.ckpt_dir = ckpt_dir

    log_dir = os.path.join(config.save_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('Log Dir:', log_dir)
    config.log_dir = log_dir
    config.log_file = os.path.join(config.log_dir, "log.txt")

    out_dir = os.path.join(config.save_dir, 'results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Out Dir:', out_dir)
    config.out_dir = out_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', type=str, default='debug')
    args = parser.parse_args()

    config = CONFIG_MAP[args.dataset]()
    
    set_seed(config.seed)
    create_output_dir(config)
    config.save(json_path=config.json_file)

    log = Log(__name__, config.log_file).getlog()

    # wandb.init(
    #     project=os.getenv("WANDB_PROJECT", "StyleFlow"),
    #     entity="609nlp",
    #     name=config.save_dir,
    # )
    # wandb.config.update(args.__dict__, allow_val_change=True)

    train_file_path_list = []
    test_file_path_list = []

    # for label in [0, 1]:
    #     train_file_path = os.path.join(config.data_dir, "align_train.{}".format(label))
    #     train_file_path_list.append(train_file_path)
    #
    #     test_file_path = os.path.join(config.data_dir, "align_test.{}".format(label))
    #     test_file_path_list.append(test_file_path)

    for label in [0, 1]:
        train_file_path = os.path.join(config.data_dir, "train.{}".format(label))
        train_file_path_list.append(train_file_path)

        test_file_path = os.path.join(config.data_dir, "test.{}".format(label))
        test_file_path_list.append(test_file_path)

    vocab = Vocab(os.path.join(config.data_dir, "vocab.txt"))

    log.info(dict((name, getattr(config, name)) for name in dir(config) if not name.startswith("__")))
    
    train_loader = get_dataloader(
        train_file_path_list,
        vocab=vocab,
        max_length=config.max_length, 
        batch_size=config.batch_size, 
        device=config.device,
        shuffle=True
    )

    test_loader = get_dataloader(
        test_file_path_list,
        vocab=vocab,
        max_length=config.max_length, 
        batch_size=config.batch_size, 
        device=config.device,
        shuffle=False
    )

    model_G = TSST(config, vocab).to(config.device)
    model_D = Discriminator(config).to(config.device)

    pretrain(config, log, vocab, model_G, train_loader, test_loader, load_ckpt=config.load_pretrain)
    train(config, log, vocab, model_G, model_D, train_loader, test_loader)

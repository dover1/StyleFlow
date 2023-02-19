import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer


class Vocab(object):
    def __init__(self, vocab_path, add_special_tokens=True):
        
        self.word2id = {}
        self.id2word = []
        # self.word_score = []

        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        for token in special_tokens:
            self.word2id[token] = len(self.word2id)
            self.id2word.append(token)
            # self.word_score.append(0)

        with open(vocab_path, "r", encoding="utf8") as f:
            for line in f:
                split_line = line.strip().split("\t")
                word = split_line[0]
                # word, freqs = split_line[0], list(map(int, split_line[1:]))
                self.word2id[word] = len(self.word2id)
                self.id2word.append(word)
                
                # # if there were only two styles, it would equal $1 - abs(freq0 - freq1) / (freq0 + freq1)$
                # score = 0
                # for freq in freqs:
                #     score += abs((freq / sum(freqs)) - (1 / len(freqs)))
                # self.word_score.append(1 - score)
        
        self.size = len(self.word2id)
        
        self.pad = self.word2id['<pad>']
        self.bos = self.word2id['<bos>']
        self.eos = self.word2id['<eos>']
        self.unk = self.word2id['<unk>']
    
    def __len__(self):
        return len(self.word2id)
    
    def special_tokens(self):
        return [self.pad, self.bos, self.eos, self.unk]


def load_json(path, label):
    print("reading %s ..." % (path))
    data = []

    with open(path, 'r', encoding='utf-8') as fin:
        results = json.load(fin)
        for dic in results:
            new_dic = {}
            new_dic['input'] = dic['x']
            new_dic['target'] = dic['y_ins'][0]
            new_dic['sim'] = dic['y_ins'][0]
            new_dic['label'] = label
            data.append(new_dic)
    return data


# Dataset
class TransferDataset(Dataset):
    def __init__(self, file_path, label):
        self.texts = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for l in lines:
                self.texts.append(l.strip())
                self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


# def collate_fn(batch, pad_id, device):
#     input_ids = []
#     attention_mask = []
#     token_type_ids = []
#     labels = []
#     texts = []
#     btc_size = len(batch)
#     max_input_len = max([len(inputs) for inputs, label, text in batch])
#
#     for btc_idx in range(btc_size):
#         inputs, label, text = batch[btc_idx]
#         input_len = len(inputs)
#         input_ids.append(inputs + [pad_id] * (max_input_len - input_len))
#
#         attention_mask.append([1] * input_len + [0] * (max_input_len - input_len))
#         token_type_ids.append([0] * max_input_len)
#         labels.append(label)
#         texts.append(text)
#
#     return {
#         "input_ids": torch.tensor(input_ids, dtype=torch.long).to(device),
#         "attention_mask": torch.tensor(attention_mask, dtype=torch.long).to(device),
#         "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).to(device),
#         "labels": torch.tensor(labels, dtype=torch.long).to(device),
#         "texts": texts,
#     }


class DataIter:
    def __init__(self, dataloaders):
        self.loaders = dataloaders
        self.per_style_size = max([len(i) for i in self.loaders]) // len(self.loaders) + 1
        self.length = self.per_style_size * len(self.loaders)
        self.init_state()
        
    def __len__(self):
        return self.length
    
    def init_state(self):
        self.iters = [iter(i) for i in self.loaders]
        self.steps = []
        for i in range(len(self.loaders)):
            self.steps += [i] * self.per_style_size
        self.cur = 0
        random.shuffle(self.steps)
        
    def __next__(self):
        if self.cur < self.length:
            label_id = self.steps[self.cur]
            for out in self.iters[label_id]:
                self.cur += 1
                return out
        else:
            self.init_state()
            raise StopIteration

    def __iter__(self):
        self.init_state()
        return self


def collate_fn(batch, vocab, tokenizer, max_length, device):
    inputs, labels = list(zip(*batch))
    input_tokens = [tokenizer(s) for s in inputs]
    max_len = min(max([len(s) for s in input_tokens]), max_length)
    input_ids = []

    for tokens in input_tokens:
        input_sent_ids = [vocab.word2id.get(t, vocab.unk) for t in tokens][:max_len]
        input_ids.append([vocab.bos] + input_sent_ids + [vocab.eos] + [vocab.pad] * (max_len - len(input_sent_ids)))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).to(device),
        "labels": torch.tensor(labels, dtype=torch.long).to(device),
        "input_texts": inputs
    }


def get_dataloader(path_list, vocab, batch_size, device, max_length=25, shuffle=False):
    datasets = []
    dataloaders = []
    cfn = lambda batch: collate_fn(batch, vocab, tokenizer=word_tokenize, max_length=max_length, device=device)
    for i, path in enumerate(path_list):
        dataset = TransferDataset(path, i)
        datasets.append(dataset)
        dataloaders.append(
            DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=cfn)
        )
    dataloader = DataIter(dataloaders)

    return dataloader

import torch
import numpy as np


def read_file(fname):
    data = []
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            data.append(line.strip())
    return data


def write_file(fname, sents):
    with open(fname, 'w') as f:
        for doc in sents:
            f.write(str(doc))
            f.write('\n')


def convert_ids_to_tokens(output_ids, vocab):
    outputs = []
    for sent in output_ids:
        tokens = []
        for word_id in sent:
            if word_id in [vocab.bos, vocab.unk, vocab.pad]:
                continue
            if word_id == vocab.eos:
                break
            else:
                tokens.append(vocab.id2word[word_id])
        outputs.append(" ".join(tokens))
    return outputs


def safe_loss(ori_loss):
    loss = torch.where(torch.isnan(ori_loss), torch.full_like(ori_loss, 0.0), ori_loss)
    loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.0), loss)
    return loss


def process_outputs(output_ids, eos_id, pad_id, sos_id):
    batch_size, _ = output_ids.size()
    sents = []
    for i in range(batch_size):
        sent = output_ids[i].cpu().tolist()
        if eos_id in sent:
            sent = sent[:sent.index(eos_id)]
        sent = [word for word in sent if word not in [eos_id, pad_id, sos_id]]
        sents.append(sent)
    
    max_len = max([len(i) for i in sents])
    inputs = [[sos_id] + sent + [eos_id] + [pad_id]*(max_len - len(sent)) for sent in sents]
    return torch.LongTensor(inputs).to(output_ids.device)


def get_bow_labels(inputs, samples, vocab):
    vocab_size = len(vocab)
    inputs_list = inputs.cpu().tolist()
    sample_list = samples.cpu().tolist()
    bow_one_hot_labels = []
    for i in range(len(inputs_list)):
        orgin_set = set(inputs_list[i])
        retrieve_set = set()
        for j in sample_list[i]:
            retrieve_set = retrieve_set | set(j)
        retrieve_set = retrieve_set - set(vocab.special_tokens())
        bow_label = list(retrieve_set - orgin_set)

        bow_label = torch.LongTensor(bow_label)
        bow_one_hot = torch.zeros(1, vocab_size)
        bow_one_hot.index_fill_(1, bow_label, 1)
        bow_one_hot_labels.append(bow_one_hot)

    bow_one_hot_labels = torch.cat(bow_one_hot_labels, dim=0).to(inputs.device)
    return bow_one_hot_labels


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

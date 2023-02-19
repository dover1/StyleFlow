import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)
        self._pad_idx = pad_idx

    # def forward(self, outputs, targets, truncate=False):
    #
    #     vocab_size = outputs.size(-1)
    #     tgts = targets.contiguous().view(-1)  # tgts: (N)
    #
    #     if truncate:
    #         tgt_len = targets.size(1)
    #         outs = outputs[:, :tgt_len, :].contiguous().view(-1, vocab_size)  # outs: (N, V)
    #     else:
    #         outs = outputs.contiguous().view(-1, vocab_size)  # outs: (N, V)
    #
    #     non_pad_mask = tgts.ne(self._pad_idx)
    #
    #     loss = self._criterion(outs, tgts)  # [N]
    #
    #     loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0.0), loss)
    #     loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.0), loss)
    #     loss = loss.masked_select(non_pad_mask).mean()
    #     return loss

    def forward(self, outputs, targets, weight=None, truncate=False):
        # outputs: (B, L, V)
        # targets: (B, L)
        # truncate: sometimes outputs may be longer than targets,
        #   we truncate outputs to the length of targets

        if truncate:
            tgt_len = targets.size(1)
            outputs = outputs[:, 0:tgt_len, :]
            outputs = outputs.transpose(1, 2)

        non_pad_mask = targets.ne(self._pad_idx)

        loss = self._criterion(outputs, targets)  # [B, L]
        if weight is not None:
            loss = loss * weight.unsqueeze(1)

        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0.0), loss)
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.0), loss)
        loss = loss.masked_select(non_pad_mask)

        return loss.mean()
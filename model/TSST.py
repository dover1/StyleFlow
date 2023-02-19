import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Layers import Encoder, Decoder
from .utils import get_seq_length, get_attn_mask
from .flows import create_normalizing_flows
from .real_nvp import FlowEncoder
from torch.distributions import MultivariateNormal


class TSST(nn.Module):
    
    def __init__(self, config, vocab):
        super().__init__()

        self.config = config
        self.vocab = vocab
        self.separate = config.separate

        self.bidirectional = config.bidirectional
        self.nlayers = config.nlayers
        self.hidden_size = config.hidden_size
        self.hidden_factor = (2 if config.bidirectional else 1) * config.nlayers

        # word embedding and weights
        self.word_embedding = nn.Embedding(len(vocab), config.embedding_size, padding_idx=vocab.pad)

        if self.separate:
            self.src_enc = Encoder(config)
            self.tgt_enc = Encoder(config)
            self.src_hidden2latent = nn.Sequential(nn.Linear(config.hidden_size * self.hidden_factor, config.hidden_size),
                                                   nn.Tanh())
            self.tgt_hidden2latent = nn.Sequential(nn.Linear(config.hidden_size * self.hidden_factor, config.hidden_size),
                                                   nn.Tanh())
            self.src_dec = Decoder(config)
            self.tgt_dec = self.src_dec
        else:
            self.src_enc = Encoder(config)
            self.tgt_enc = self.src_enc
            self.src_hidden2latent = nn.Sequential(
                nn.Linear(config.hidden_size * self.hidden_factor, config.hidden_size),
                nn.Tanh())
            self.tgt_hidden2latent = self.src_hidden2latent
            self.src_dec = Decoder(config)
            self.tgt_dec = self.src_dec

        self.bidirectional = config.bidirectional
        self.nlayers = config.nlayers
        self.hidden_size = config.hidden_size
        self.hidden_factor = (2 if config.bidirectional else 1) * config.nlayers

        self.h2word = nn.Linear(config.hidden_size, len(vocab))

        # flow_kwargs = {'hiddenflow_layers': config.hiddenflow_layers, 'flow_units': config.hidden_size,
        #                'flow_nums': config.flow_nums}
        # self.src_flow = create_normalizing_flows(flow_type=config.flow_type, z_size=config.hidden_size,
        #                                          dropout=0, kwargs=flow_kwargs)
        # self.tgt_flow = create_normalizing_flows(flow_type=config.flow_type, z_size=config.hidden_size,
        #                                          dropout=0, kwargs=flow_kwargs)
        self.use_flow = config.use_flow
        if self.use_flow:
            base_mu, base_cov = torch.zeros(config.hidden_size).to(config.device), torch.eye(config.hidden_size).to(
                config.device)
            self.base_dist = MultivariateNormal(base_mu, base_cov)
            self.src_flow = FlowEncoder(self.base_dist, config.hidden_size, config.flow_units,
                                        k=config.flow_nums)
            self.tgt_flow = FlowEncoder(self.base_dist, config.hidden_size, config.flow_units,
                                        k=config.flow_nums)
        else:
            self.src_flow = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.Tanh())
            self.tgt_flow = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.Tanh())

    def get_word_embedding(self, inputs):
        if inputs.dim() == 3:
            return torch.matmul(F.softmax(inputs, dim=-1), self.word_embedding.weight)
        else:
            return self.word_embedding(inputs)
    
    def get_encode_state(self, enc_state):
        bsz = enc_state[0].size(1)
        if self.config.bidirectional:
            enc_state_h = enc_state[0][-2:].transpose(0, 1).contiguous().view(bsz, -1)
            enc_state_c = enc_state[1][-2:].transpose(0, 1).contiguous().view(bsz, -1)
        else:
            enc_state_h = enc_state[0][-1]
            enc_state_c = enc_state[1][-1]
        return enc_state_h, enc_state_c

    def init_dec_state(self, enc_state):

        enc_state_h, enc_state_c = self.get_encode_state(enc_state)

        init_state_h = self.enc2dec_h(enc_state_h).unsqueeze(0)
        # init_state_c = self.enc2dec_c(enc_state_c).unsqueeze(0)
        init_state_c = torch.zeros_like(init_state_h)

        init_state = [
            init_state_h.repeat(self.config.nlayers, 1, 1),
            init_state_c.repeat(self.config.nlayers, 1, 1)
        ]

        return init_state

    def encode(self, inputs, encoder, hidden2latent):
        batch_size = inputs.size(0)
        lengths = get_seq_length(inputs, self.vocab.pad)
        input_embeds = self.word_embedding(inputs)
        enc_outs, hidden = encoder(input_embeds, lengths)
        if self.bidirectional or self.nlayers > 1:
            # flatten hidden state
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        latent = hidden2latent(hidden)
        return enc_outs, latent
    
    def decode_step(self, input, last_state, enc_outs, attn_mask, decoder):
        input_embed = self.word_embedding(input)
        out, state, _ = decoder(input_embed, last_state, enc_outs, attn_mask)
        logits = self.h2word(out)
        return logits, state

    def decode(self, enc_outs, latent, attn_mask, decoder, targets=None):
        batch_size = latent.size(0)
        last_state = latent.unsqueeze(0)

        if targets is not None:
            target_len = targets.size(1)
        
        output_logits = []
        input = torch.zeros(batch_size, 1, dtype=torch.long, device=last_state.device).fill_(self.vocab.bos)

        for t in range(self.config.max_length+1):
            logits, last_state = self.decode_step(input, last_state, enc_outs, attn_mask, decoder)
            output_logits.append(logits.unsqueeze(1))            
            is_teacher = random.random() < self.config.teacher_forcing_ratio
            
            if (targets is not None) and (t < target_len) and is_teacher:
                input = targets[:, t].unsqueeze(1)
            else:
                input = torch.argmax(logits, dim=-1).unsqueeze(1)
        
        output_logits = torch.cat(output_logits, dim=1)
        return output_logits
    
    def forward(self, inputs, labels, targets):

        attn_mask = get_attn_mask(inputs, self.vocab.pad)
        if labels[0] == 0:
            enc_outs, latent = self.encode(inputs, self.src_enc, self.src_hidden2latent)
            x = latent
            if self.use_flow:
                z, log_p = self.src_flow.forward(x.detach().clone())  # [B]

            output_logits = self.decode(enc_outs, latent, attn_mask, self.src_dec, targets)
        else:
            enc_outs, latent = self.encode(inputs, self.tgt_enc, self.tgt_hidden2latent)
            x = latent
            if self.use_flow:
                z, log_p = self.tgt_flow.forward(x.detach().clone())  # [B]

            output_logits = self.decode(enc_outs, latent, attn_mask, self.tgt_dec, targets)

        if self.use_flow:
            nf_loss = -log_p.mean()
        else:
            nf_loss = torch.tensor([0]).to(inputs.device)
        return output_logits, nf_loss

    def sample(self, num_samples=1):
        z = self.base_dist.sample((num_samples, ))

        src_h, _ = self.src_flow.inverse(z)  # [B]
        dec_state = src_h
        src_logits = self.decode(None, dec_state, None, self.src_dec, None)

        tgt_h, _ = self.tgt_flow.inverse(z)  # [B]
        dec_state = tgt_h
        tgt_logits = self.decode(None, dec_state, None, self.tgt_dec, None)

        return src_logits, tgt_logits

    def transfer_forward(self, inputs, labels, targets):

        attn_mask = get_attn_mask(inputs, self.vocab.pad)
        if labels[0] == 0:
            enc_outs, latent = self.encode(inputs, self.src_enc, self.src_hidden2latent)
            x = latent
            if self.use_flow:
                z, log_p = self.src_flow.forward(x)  # [B]
                trans_x, log_p = self.tgt_flow.inverse(z)
            else:
                z =  torch.tensor([0]).to(inputs.device)
                trans_x = self.tgt_flow(x)

            output_logits = self.decode(enc_outs, trans_x, attn_mask, self.tgt_dec, targets)
        else:
            enc_outs, latent = self.encode(inputs, self.tgt_enc, self.tgt_hidden2latent)
            x = latent
            if self.use_flow:
                z, log_p = self.tgt_flow.forward(x)  # [B]
                trans_x, log_p = self.src_flow.inverse(z)
            else:
                z =  torch.tensor([0]).to(inputs.device)
                trans_x = self.src_flow(x)

            output_logits = self.decode(enc_outs, trans_x, attn_mask, self.src_dec, targets)

        return output_logits, x, z, trans_x

    def cycle_forward(self, inputs, labels, targets):

        attn_mask = get_attn_mask(inputs, self.vocab.pad)
        if labels[0] == 0:
            enc_outs, latent = self.encode(inputs, self.src_enc, self.src_hidden2latent)
            x = latent
            z, log_p = self.src_flow.forward(x)  # [B]
            trans_x, log_p = self.tgt_flow.inverse(z)
            z, log_p = self.tgt_flow.forward(trans_x)
            cyc_x, log_p = self.src_flow.inverse(z)

            output_logits = self.decode(enc_outs, cyc_x, attn_mask, self.src_dec, targets)
        else:
            enc_outs, latent = self.encode(inputs, self.tgt_enc, self.tgt_hidden2latent)
            x = latent
            z, log_p = self.tgt_flow.forward(x)  # [B]
            trans_x, log_p = self.src_flow.inverse(z)
            z, log_p = self.src_flow.forward(trans_x)
            cyc_x, log_p = self.tgt_flow.inverse(z)

            output_logits = self.decode(enc_outs, cyc_x, attn_mask, self.tgt_dec, targets)
        # pdist = nn.PairwiseDistance(p=2)
        # dist = pdist(x, cycle_x).mean() * self.config.dist_factor

        return output_logits

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from config import LABEL_MAP
from model import Criterion, ISRScheduler
from utils import convert_ids_to_tokens, safe_loss, process_outputs, get_bow_labels
from evaluator import Evaluator
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import collections


def geometric_mean(data):  # 计算几何平均数
    total=1
    for i in data:
        total*=i #等同于total=total*i
    return pow(total,1/len(data))


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def evaluate_bleu(config, model_G, dev_dataloader, vocab, epoch):
    with torch.no_grad():
        for loader in dev_dataloader.loaders:
            inp_outputs = []
            rec_outputs = []
            for batch in loader:
                inputs, labels = batch['input_ids'], batch['labels']
                rec_outs, _ = model_G.forward(inputs, labels, targets=None)
                rec_outs = rec_outs.argmax(dim=-1)
                inp_outputs += convert_ids_to_tokens(inputs, vocab)
                rec_outputs += convert_ids_to_tokens(rec_outs, vocab)

    smooth = SmoothingFunction()
    bleu = corpus_bleu([[r] for r in inp_outputs], rec_outputs, smoothing_function=smooth.method1)
    bleu = np.round(bleu * 100, 2)

    return bleu


def save_pretrain_checkpoint(ckpt_path, model):
    # save model state dict
    saved_dic = collections.OrderedDict()
    keysNeedStored = ['word_embedding', 'src_enc', 'tgt_enc', 'src_dec', 'tgt_dec',
                      'src_hidden2latent', 'tgt_hidden2latent',
                      'src_latent2hidden', 'tgt_latent2hidden',
                      'src_flow', 'tgt_flow',
                      'h2word']
    for key in model.state_dict().keys():
        if key.split(".")[0] in keysNeedStored:
            saved_dic[key] = model.state_dict()[key]
    torch.save(saved_dic, ckpt_path)


def load_pretrain_checkpoint(ckpt_path, model, log):
    pre_state_dict = torch.load(ckpt_path)
    gen_state_dict = model.state_dict()

    overlap_dic = {}
    for p_key in pre_state_dict.keys():
        for g_key in model.state_dict().keys():
            if p_key == g_key:
                overlap_dic[g_key] = pre_state_dict[p_key]
                log.info("load %s from %s" % (g_key, p_key))

    gen_state_dict.update(overlap_dic)
    model.load_state_dict(gen_state_dict)
    log.info("Loading pretrained params done!")


def train_nf(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion):
    inputs, labels = batch['input_ids'], batch['labels']

    # rec loss
    rec_logits, nf_loss = model_G.forward(inputs, labels, targets=None)
    rec_loss = seq_criterion(rec_logits, inputs[:, 1:], truncate=True) * config.rec_factor
    nf_loss = nf_loss * config.nf_factor
    losses = rec_loss + nf_loss
    losses = losses.mean()
    if not torch.isnan(losses).item():
        optim_G.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), config.max_grad_norm)
        optim_G.step()

    return rec_loss.item(), nf_loss.item()


def train_cycle(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion):
    inputs, labels = batch['input_ids'], batch['labels']

    # cyc loss
    cyc_logits = model_G.cycle_forward(inputs, labels, targets=None)
    cyc_loss = seq_criterion(cyc_logits, inputs[:, 1:], truncate=True)
    # cyc_loss = torch.tensor([0]).to(inputs.device)

    losses = cyc_loss
    losses = losses.mean()
    if not torch.isnan(losses).item():
        optim_G.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), config.max_grad_norm)
        optim_G.step()

    return cyc_loss.item()


def train_generator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion):
    inputs, labels = batch['input_ids'], batch['labels']

    # rec loss
    rec_logits, nf_loss = model_G.forward(inputs, labels, targets=inputs[:, 1:])
    rec_loss = seq_criterion(rec_logits, inputs[:, 1:], truncate=True) * config.rec_factor
    nf_loss = nf_loss * config.nf_factor
    # rec_loss = torch.tensor([0]).to(inputs.device)

    # cyc loss
    rev_logits, _, _, _ = model_G.transfer_forward(inputs, labels, targets=None)
    rev_inputs = process_outputs(torch.argmax(rev_logits, dim=-1), vocab.eos, vocab.pad, vocab.bos)
    cyc_logits, _, _, _ = model_G.transfer_forward(rev_inputs, 1-labels, targets=inputs[:, 1:])
    cyc_loss = seq_criterion(cyc_logits, inputs[:, 1:], truncate=True) * config.cyc_factor

    # adv loss
    rev_input_embeds = model_G.get_word_embedding(rev_logits)
    class_logits = model_D(rev_input_embeds)
    adv_loss = cls_criterion(class_logits, 1-labels) * config.adv_factor
    # adv_loss = torch.tensor([0]).to(inputs.device)

    losses = rec_loss + cyc_loss + adv_loss + nf_loss
    losses = losses.mean()
    if not torch.isnan(losses).item():
        optim_G.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), config.max_grad_norm)
        optim_G.step()
    
    return losses.item(), rec_loss.item(), cyc_loss.item(), adv_loss.item(), nf_loss.item(), 0


def train_disriminator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion):
    
    inputs, labels = batch['input_ids'], batch['labels']

    with torch.no_grad():
        rec_logits, _ = model_G.forward(inputs, labels, targets=inputs[:, 1:])
        rev_logits, _, _, _ = model_G.transfer_forward(inputs, labels, targets=None)
        rev_input_embeds = model_G.get_word_embedding(rev_logits)
        raw_input_embeds = model_G.get_word_embedding(rec_logits)
        golden_input_embedds = model_G.get_word_embedding(inputs[:, 1:])
    
    rev_loss = cls_criterion(model_D(rev_input_embeds), torch.zeros_like(labels).fill_(2))
    rec_loss = cls_criterion(model_D(raw_input_embeds), labels)
    raw_loss = cls_criterion(model_D(golden_input_embedds), labels)
    
    dis_loss = raw_loss + rec_loss + rev_loss
    
    optim_D.zero_grad()
    dis_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_D.parameters(), config.max_grad_norm)
    optim_D.step()
    return dis_loss.item()


def pretrain(config, log, vocab, model_G, train_dataloader, dev_dataloader, load_ckpt):
    ckpt_path = os.path.join(config.ckpt_dir, 'pretrain_model.pt')

    if load_ckpt:
        try:
            # load_pretrain_checkpoint(ckpt_path, model_G, log)
            model_G.load_state_dict(torch.load(ckpt_path))
            return
        except Exception as e:
            log.info("Loading pretrained lm model error: {}".format(e))
            pass

    log.info("Begin pretraining ...")

    optim_G = torch.optim.Adam(model_G.parameters(), lr=0.0, betas=(0.9, 0.999))
    optim_G = ISRScheduler(optimizer=optim_G, warmup_steps=config.warmup_steps,
                           max_lr=config.max_lr, min_lr=config.min_lr, init_lr=config.init_lr, beta=0.75)
    seq_criterion = Criterion(vocab.pad)

    # pretrain
    model_G.train()
    best_result = 0
    epoch = 0
    patience = 0
    global_step = 0
    for epoch in range(1, config.pretrain_epochs+1):
        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch['input_ids'], batch['labels']
            KL_weight = kl_anneal_function(config.anneal_function, global_step, config.k, config.x0)
            # rec loss
            rec_logits, nf_loss = model_G.forward(inputs, labels, targets=inputs[:, 1:])
            rec_loss = seq_criterion(rec_logits, inputs[:, 1:], truncate=True)
            nf_loss = nf_loss * config.nf_factor * KL_weight
            loss = rec_loss + nf_loss

            optim_G.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_G.parameters(), config.max_grad_norm)
            optim_G.step()
            global_step += 1
            if step % 20 == 0:
                log.info("Pretrain Epoch {}, step {}/{}, rec_loss: {:.2f}, nf_loss: {:.2f}, lr_G: {:.7f}".format(
                        epoch, step + 1, len(train_dataloader), rec_loss.item(), nf_loss.item(), optim_G.rate()))

        result = evaluate_bleu(config, model_G, dev_dataloader, vocab, epoch)
        print("best blue: %.4f, current blue: %.4f" % (best_result, result))
        if best_result < result:
            best_result = result
            print("Saving model checkpoint to %s" % ckpt_path)
            save_pretrain_checkpoint(ckpt_path, model_G)
            patience = 0
        else:
            patience += 1
        if patience > 3:
            break


def train(config, log, vocab, model_G, model_D, train_dataloader, dev_dataloader):
    writer = SummaryWriter(config.out_dir)

    # need_frozen_list 需要更新梯度的变量
    # need_update_list = ['src_flow', 'tgt_flow']
    # for param in model_G.named_parameters():
    #     if param[0].split('.')[0] not in need_update_list:
    #         param[1].requires_grad = False
    # optim_G = torch.optim.Adam(filter(lambda p: p.requires_grad, model_G.parameters()), lr=0.0, betas=(0.9, 0.999))

    optim_G = torch.optim.Adam(model_G.parameters(), lr=0.0, betas=(0.9, 0.999))
    optim_D = torch.optim.Adam(model_D.parameters(), lr=0.0, betas=(0.5, 0.999))

    optim_G = ISRScheduler(optimizer=optim_G, warmup_steps=config.warmup_steps,
                           max_lr=config.max_lr, min_lr=config.min_lr, init_lr=config.init_lr, beta=0.75)

    optim_D = ISRScheduler(optimizer=optim_D, warmup_steps=config.warmup_steps,
                           max_lr=config.max_lr, min_lr=config.min_lr, init_lr=config.init_lr, beta=0.5)

    seq_criterion = Criterion(vocab.pad)
    cls_criterion = Criterion(-1)

    # useFlow = config.use_flow
    # config.use_flow = False
    # only different dec for transfer
    auto_eval(config, vocab, model_G, dev_dataloader, 0, writer)
    # config.use_flow = useFlow

    # train
    global_steps = 0
    for epoch in range(1, config.epochs+1):
        for step, batch in enumerate(train_dataloader):

            model_D.train()
            model_G.train()

            # train generator
            losses, rec_loss, cyc_loss, adv_loss, nf_loss, dist_loss = \
                train_generator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion)

            # rec_loss, nf_loss = train_nf(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion,
            #                              cls_criterion)
            #
            # rev_loss = train_transfer(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion,
            #                           cls_criterion)
            #
            # cyc_loss = train_cycle(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion,
            #                        cls_criterion)

            # train disriminator
            dis_loss = 0
            for _ in range(config.d_step):
                dis_loss = train_disriminator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion)

            if global_steps % 20 == 0:
                log.info(
                    "epoch {}, step {}/{}, rec_loss: {:.2f}, cyc_loss: {:.2f}, adv_loss: {:.2f}, nf_loss: {:.2f}, dis_loss: {:.2f}, lr_G: {:.7f}, lr_D:{:.7f}".format(
                    epoch, step+1, len(train_dataloader), rec_loss, cyc_loss, adv_loss, nf_loss, dis_loss, optim_G.rate(), optim_D.rate()
                ))

            global_steps += 1
        # log.info("saving epoch {}...".format(epoch+1))
        # torch.save(model_G, os.path.join(config.ckpts_dir, "G_{}.pt".format(epoch+1)))
        # torch.save(model_D, os.path.join(config.ckpts_dir, "D_{}.pt".format(epoch+1)))

        auto_eval(config, vocab, model_G, dev_dataloader, epoch, writer)
    return


def auto_eval(config, vocab, model_G, dev_dataloader, epoch, writer):
    model_G.eval()
    c_x_results = []
    z_results = []
    trans_x_results = []
    inp_results = []
    rec_results = []
    rev_results = []
    with torch.no_grad():
        for loader in dev_dataloader.loaders:
            c_x_outputs = []
            z_outputs = []
            trans_x_outputs = []
            inp_outputs = []
            rec_outputs = []
            rev_outputs = []
            for batch in loader:
                inputs, labels = batch['input_ids'], batch['labels']
                rec_outs, _ = model_G.forward(inputs, labels, targets=None)
                rec_outs = rec_outs.argmax(dim=-1)
                rev_outs, c_x, z, trans_x = model_G.transfer_forward(inputs, labels, targets=None)
                rev_outs = rev_outs.argmax(dim=-1)
                c_x_outputs.append(c_x)
                z_outputs.append(z)
                trans_x_outputs.append(trans_x)
                inp_outputs += convert_ids_to_tokens(inputs, vocab)
                rec_outputs += convert_ids_to_tokens(rec_outs, vocab)
                rev_outputs += convert_ids_to_tokens(rev_outs, vocab)
            c_x_results.append(torch.concat(c_x_outputs, dim=0))
            z_results.append(torch.concat(z_outputs, dim=0))
            trans_x_results.append(torch.concat(trans_x_outputs, dim=0))
            inp_results.append(inp_outputs)
            rec_results.append(rec_outputs)
            rev_results.append(rev_outputs)

    for i, _ in enumerate(zip(c_x_results, z_results, trans_x_results)):
        torch.save(c_x_results[i], os.path.join(config.out_dir, "epoch_{}_c_x.{}".format(epoch, LABEL_MAP[config.dataset][i])))
        torch.save(z_results[i], os.path.join(config.out_dir, "epoch_{}_z.{}".format(epoch, LABEL_MAP[config.dataset][i])))
        torch.save(c_x_results[i], os.path.join(config.out_dir, "epoch_{}_trans_x.{}".format(epoch, LABEL_MAP[config.dataset][i])))

    if config.use_flow:
        sample_neg_outs, sample_pos_outs = model_G.sample(num_samples=100)
        sample_neg_outs = sample_neg_outs.argmax(dim=-1)
        sample_neg_outputs = convert_ids_to_tokens(sample_neg_outs, vocab)
        sample_pos_outs = sample_pos_outs.argmax(dim=-1)
        sample_pos_outputs = convert_ids_to_tokens(sample_pos_outs, vocab)

        for i, res in enumerate([sample_neg_outputs, sample_pos_outputs]):
            with open(os.path.join(config.out_dir, "epoch_{} sample.{}".format(epoch, LABEL_MAP[config.dataset][i])), "w",
                  encoding="utf8") as f:
                for line in res:
                    f.write(line + "\n")

    for i, res in enumerate(rev_results):
        with open(os.path.join(config.out_dir, "epoch_{}.{}".format(epoch, LABEL_MAP[config.dataset][i])), "w",
                  encoding="utf8") as f:
            for line in res:
                f.write(line + "\n")

    evaluator = Evaluator(config)
    ref_text = evaluator.ref
    if ref_text is None:
        ref_text = [[[j] for j in i] for i in inp_results]
    transfered_sents = rev_results[0] + rev_results[1]
    labels = [1] * len(rev_results[0]) + [0] * len(rev_results[1])
    eval_str, (acc, sbleu, rbleu, ppl, gm) = evaluator.evaluate(transfered_sents, labels)
    print(eval_str)
    # acc_neg = evaluator.get_acc(rev_results[0], [1]*len(rev_results[0]))
    # acc_pos = evaluator.get_acc(rev_results[1], [0]*len(rev_results[0]))
    # if ref_text is not None:
    #     ref_bleu_neg = evaluator.get_bleu(ref_text[0], rev_results[0])
    #     ref_bleu_pos = evaluator.get_bleu(ref_text[1], rev_results[1])
    # else:
    #     ref_text = inp_results
    #     ref_bleu_neg = -1
    #     ref_bleu_pos = -1
    # slf_bleu_neg = evaluator.get_bleu([[i] for i in inp_results[0]], rev_results[0])
    # slf_bleu_pos = evaluator.get_bleu([[i] for i in inp_results[1]], rev_results[1])
    # ppl_neg = evaluator.get_ppl(rev_results[0])
    # ppl_pos = evaluator.get_ppl(rev_results[1])
    # acc = (acc_neg + acc_pos) / 2
    # sbleu = (slf_bleu_neg + slf_bleu_pos) / 2
    # rbleu = (ref_bleu_neg + ref_bleu_pos) / 2
    # ppl = (ppl_neg + ppl_pos) / 2
    # def geometric_mean(data):  # 计算几何平均数
    #     total = 1
    #     for i in data:
    #         total *= i  # 等同于total=total*i
    #     return pow(total, 1 / len(data))
    #
    # gm = geometric_mean([acc, sbleu, rbleu, 1 / math.log(ppl)])

    writer.add_scalar('ACC', acc, epoch)
    writer.add_scalar('SBLEU', sbleu, epoch)
    writer.add_scalar('RBLEU', rbleu, epoch)
    writer.add_scalar('PPL', ppl, epoch)
    writer.add_scalar('GM', gm, epoch)
    # metrics = {'ACC': (acc_neg + acc_pos) / 2,
    #            'SBLEU': (slf_bleu_neg + slf_bleu_pos) / 2,
    #            'RBLEU': (ref_bleu_neg + ref_bleu_pos) / 2,
    #            'PPL': (ppl_neg + ppl_pos) / 2
    #            }
    # wandb.log(metrics)

    # print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
    #        'sbleu_pos: {:.4f} sbleu_neg: {:.4f} ' + \
    #        'rbleu_pos: {:.4f} rbleu_neg: {:.4f} ' + \
    #        'ppl_pos: {:.4f} ppl_neg: {:.4f} gm: {:.4f}\n').format(
    #     acc_pos, acc_neg, slf_bleu_pos, slf_bleu_neg, ref_bleu_pos, ref_bleu_neg, ppl_pos, ppl_neg, gm
    # ))

    # save output
    save_file = config.out_dir + '/comp_' + str(epoch) + '.txt'
    eval_log_file = config.out_dir + '/eval_log.txt'
    with open(eval_log_file, 'a') as fl:
        # print(('iter{:5d}:  acc_pos: {:.4f} acc_neg: {:.4f} ' + \
        #        'sbleu_pos: {:.4f} sbleu_neg: {:.4f} ' + \
        #        'rbleu_pos: {:.4f} rbleu_neg: {:.4f} ' + \
        #        'ppl_pos: {:.4f} ppl_neg: {:.4f} gm: {:.4f}\n').format(
        #     epoch, acc_pos, acc_neg, slf_bleu_pos, slf_bleu_neg, ref_bleu_pos, ref_bleu_neg, ppl_pos, ppl_neg, gm
        # ), file=fl)
        print(eval_str, file=fl)
    with open(save_file, 'w') as fw:
        # print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
        #        'sbleu_pos: {:.4f} sbleu_neg: {:.4f} ' + \
        #        'rbleu_pos: {:.4f} rbleu_neg: {:.4f} ' + \
        #        'ppl_pos: {:.4f} ppl_neg: {:.4f} gm: {:.4f}\n').format(
        #     acc_pos, acc_neg, slf_bleu_pos, slf_bleu_neg, ref_bleu_pos, ref_bleu_neg, ppl_pos, ppl_neg,gm
        # ), file=fw)
        print(eval_str, file=fw)

        for idx in range(len(rev_results[0])):
            print('*' * 20, f'{LABEL_MAP[config.dataset][0]} sample', '*' * 20, file=fw)
            print('[inp ]', inp_results[0][idx], file=fw)
            print('[rec ]', rec_results[0][idx], file=fw)
            print('[rev ]', rev_results[0][idx], file=fw)
            print('[ref ]', '\n       '.join(ref_text[0][idx]), file=fw)

        for idx in range(len(rev_results[1])):
            print('*' * 20, f'{LABEL_MAP[config.dataset][1]} sample', '*' * 20, file=fw)
            print('[inp ]', inp_results[1][idx], file=fw)
            print('[rec ]', rec_results[1][idx], file=fw)
            print('[rev ]', rev_results[1][idx], file=fw)
            print('[ref ]', '\n       '.join(ref_text[1][idx]), file=fw)

    model_G.train()

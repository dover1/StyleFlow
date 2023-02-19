from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

import os
import fasttext
import argparse
import kenlm
import math

from config import CONFIG_MAP, LABEL_MAP
from utils import read_file


class Evaluator(object):

    def __init__(self, config):
        self.dataset = config.dataset

        # original data and references for self-bleu and ref-bleu
        self.data_dir = config.data_dir
        self.ori_data = self.get_data(file_name="test")
        self.ref_data = self.get_data(file_name="ref")

        if config.dataset in ['debug', 'yelp', 'amazon', 'shakespeare', 'caption']:
            ref0_path = f'{config.raw_data_dir}/ref.0'
            ref1_path = f'{config.raw_data_dir}/ref.1'

            self.ref = []
            with open(ref0_path, 'r') as fin:
                self.ref.append([[l.strip()] for l in fin.readlines()])
            with open(ref1_path, 'r') as fin:
                self.ref.append([[l.strip()] for l in fin.readlines()])
        elif config.dataset in ['gyafc']:
            ref0_paths = [f'{config.raw_data_dir}/ref.0.{i}' for i in range(4)]
            ref1_paths = [f'{config.raw_data_dir}/ref.1.{i}' for i in range(4)]

            ref = []
            for i in range(4):
                with open(ref0_paths[i], 'r') as fin:
                    ref.append([l.strip() for l in fin.readlines()])
            ref_0 = []
            for j in range(len(ref[0])):
                ref_0.append([ref[i][j] for i in range(4)])

            ref = []
            for i in range(4):
                with open(ref1_paths[i], 'r') as fin:
                    ref.append([l.strip() for l in fin.readlines()])
            ref_1 = []
            for j in range(len(ref[0])):
                ref_1.append([ref[i][j] for i in range(4)])
            self.ref = []
            self.ref.append(ref_0)
            self.ref.append(ref_1)
        elif config.dataset in ['cpvt']:
            self.ref = None
            ref0_path = f'{config.prepro_data_dir}/test.1'
            ref1_path = f'{config.prepro_data_dir}/test.0'

            self.ref = []
            with open(ref0_path, 'r') as fin:
                self.ref.append([[l.strip()] for l in fin.readlines()])
            with open(ref1_path, 'r') as fin:
                self.ref.append([[l.strip()] for l in fin.readlines()])
        else:
            self.ref = None

        self.classifier = fasttext.load_model(config.classifier_path)
        # self.ppl_model = kenlm.Model(config.ppl_path)
        # language model for ppl
        self.lm_model = []
        for label in LABEL_MAP[self.dataset]:
            lm_path = os.path.join('/home/hyh/PycharmProjects/nf/evaluator/kenlm', "{}.{}.bin".format(self.dataset, label))
            self.lm_model.append(kenlm.LanguageModel(lm_path))

    def get_data(self, file_name="test"):
        data = []
        if (self.dataset == "politeness" or self.dataset == "imdb") and file_name == "ref":
            return None
        for label in LABEL_MAP[self.dataset]:
            if self.dataset == "gyafc" and file_name == "ref":
                gyafc_ref = [read_file(os.path.join(self.data_dir, "{}.{}.{}".format(file_name, label, i))) for i in range(4)]
                label_data = list(zip(gyafc_ref[0], gyafc_ref[1], gyafc_ref[2], gyafc_ref[3]))
                data += [[sent for sent in sents] for sents in label_data]
            else:
                label_data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
                data += [[sent] for sent in label_data]
        return data

    def get_acc(self, transfered_sents, labels):
        sents = [t if t != '' else '1' for t in transfered_sents]
        preds = self.classifier.predict(sents)
        preds = [l == ['__label__one'] for l in preds]
        total_count = len(transfered_sents)
        right_count = 0
        for i in range(total_count):
            if preds[i] == labels[i]:
                right_count += 1

        return right_count / total_count * 100

    def get_bleu(self, ref_sents, transfered_sents):
        try:
            assert len(transfered_sents) == len(ref_sents)
        except:
            print(len(transfered_sents))
        ref_seg_sents = [[sent.split() for sent in sents] for sents in ref_sents]
        transfered_seg_sents = [sent.split() for sent in transfered_sents]
        return corpus_bleu(ref_seg_sents, transfered_seg_sents) * 100

    # def get_ppl(self, transfered_sents):
    #     texts = [' '.join(word_tokenize(itm.lower().strip())) for itm in transfered_sents]
    #     sum = 0
    #     words = []
    #     length = 0
    #     for i, line in enumerate(texts):
    #         words += [word for word in line.split()]
    #         length += (len(line.split()) + 1)
    #         score = self.ppl_model.score(line)
    #         sum += score
    #     return math.pow(10, -sum / length)
    def get_ppl(self, sents, labels):
        total_score = 0
        total_length = 0
        for sent, label in zip(sents, labels):
            total_score += self.lm_model[label].score(sent)
            total_length += len(sent.split())

        if total_length == 0:
            print(total_score, total_length)
            return math.pow(10, 4)
        else:
            return math.pow(10, -total_score / (total_length))

    def evaluate(self, transfered_sents, labels):

        acc = self.get_acc(transfered_sents, labels)
        ppl = self.get_ppl(transfered_sents, labels)

        self_bleu = self.get_bleu(self.ori_data, transfered_sents)
        ref_bleu = self.get_bleu(self.ref_data, transfered_sents) if self.ref_data is not None else 0

        # gm = math.pow(acc * self_bleu * ref_bleu * 1.0 / math.log(ppl), 1.0 / 4.0)
        gm = math.pow(acc * self_bleu * 1.0 / math.log(ppl), 1.0 / 3.0)

        eval_str = "ACC: {:.2f} \tself-BLEU: {:.2f} \tref-BLEU: {:.2f} \tPPL: {:.2f} \tGM: {:.2f}".format(acc,
                                                                                                          self_bleu,
                                                                                                          ref_bleu, ppl,
                                                                                                          gm)
        return eval_str, (acc, self_bleu, ref_bleu, ppl, gm)

    def evaluate_model(self, result_file):
        transfered_sents = []
        labels = []
        result_file = os.path.join(result_file, 'out')
        for i, label in enumerate(LABEL_MAP[self.dataset]):
            sents = read_file("{}.{}".format(result_file, label))
            transfered_sents += sents
            labels += [1 - i] * len(sents)

        eval_str, (acc, self_bleu, ref_bleu, ppl, gm) = self.evaluate(transfered_sents, labels)
        return eval_str, (acc, self_bleu, ref_bleu, ppl, gm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gyafc")
    parser.add_argument('--datadir', type=str, default="outputs")
    parser.add_argument('--file', type=str, default="all")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Evaluating dataset: {dataset_name}")

    if args.file == "all":
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.join(cur_dir, args.datadir), dataset_name)
        result_models = set()
        for model_name in os.listdir(dataset_dir):
            result_models.add(os.path.join(dataset_dir, model_name))
    else:
        result_models = [args.file]

    config = CONFIG_MAP[args.dataset]()
    evaluator = Evaluator(config)

    ref_acc, ref_self_bleu, ref_ref_bleu, ref_ppl, ref_gm = 0, 0, 0, 0, 0
    ref_num = 0
    for result_model in result_models:
        print(result_model, end="\t")
        eval_str, (acc, self_bleu, ref_bleu, ppl, gm) = evaluator.evaluate_model(result_model)
        if 'Reference' in result_model:
            ref_num += 1
            ref_acc += acc
            ref_self_bleu += self_bleu
            ref_ref_bleu += ref_bleu
            ref_ppl += ppl
            ref_gm += gm
        print(eval_str)

    print('Reference', end="\t")
    print("ACC: {:.2f} \tself-BLEU: {:.2f} \tref-BLEU: {:.2f} \tPPL: {:.2f} \tGM: {:.2f}".format(ref_acc / ref_num,
                                                                                                 ref_self_bleu / ref_num,
                                                                                                 ref_ref_bleu / ref_num,
                                                                                                 ref_ppl / ref_num,
                                                                                                 ref_gm / ref_num,
                                                                                                 ))
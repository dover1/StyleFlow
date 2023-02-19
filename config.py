import json


class Config():
    def __init__(self):
        pass

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def load(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        # model
        self.seed = 6
        self.embedding_size = 768
        self.nlayers = 1
        self.bidirectional = True
        self.hidden_size = 768
        self.max_length = 18
        self.batch_size = 128
        self.num_filters = 64

        # data
        self.dataset = "debug"
        self.raw_data_dir = './data/debug/raw'
        self.data_dir = './data/debug/prepro'
        self.save_dir = "./save"
        self.pretrain_ckpt_dir = './save/debug/pretrain_ckpt'
        self.classifier_path = '/home/hyh/PycharmProjects/few-shot/classification/yelp/fasttext/fasttext_classifier.bin'
        self.ppl_path = '/home/hyh/PycharmProjects/few-shot/dataset/yelp/proc/ppl_all.bin'

        # flows related
        self.anneal_function = 'logistic'
        self.k = 0.0025
        self.x0 = 2000
        self.use_flow = False
        self.separate = True
        self.flow_type = 'realNVP' if self.use_flow else None  # The type of flow to be used, none, scf or glow
        self.flow_nums = 1 if self.use_flow else None  # Number of flows in the NF
        self.flow_units = [self.hidden_size // 2, self.hidden_size // 4, self.hidden_size // 2]  # Size of MLP in scf flows

        # loss factor
        self.nf_factor = 0.01
        self.rec_factor = 1
        self.rev_factor = 1
        self.cyc_factor = 1
        self.adv_factor = 1
        self.dist_factor = 1

        # pretrain
        self.load_pretrain = True
        self.pretrain_epochs = 10

        # train
        self.d_step = 1
        self.warmup_steps = 3000
        self.max_lr = 8e-4
        self.min_lr = 5e-5
        self.init_lr = 1e-5

        self.epochs = 10
        self.max_grad_norm = 5.0

        self.teacher_forcing_ratio = 0.0
        self.attention_dropout = 0.1
        self.dropout = 0.1


class YelpConfig(Config):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        # model
        self.seed = 11
        self.embedding_size = 768
        self.nlayers = 1
        self.bidirectional = True
        self.hidden_size = 768
        self.max_length = 18
        self.batch_size = 128
        self.num_filters = 64

        # data
        self.dataset = "yelp"
        self.raw_data_dir = './data/yelp/raw'
        self.data_dir = './data/yelp/prepro'
        self.save_dir = "./save"
        self.pretrain_ckpt_dir = './save/yelp/pretrain_ckpt'
        self.classifier_path = '/home/hyh/PycharmProjects/few-shot/classification/yelp/fasttext/fasttext_classifier.bin'
        self.ppl_path = '/home/hyh/PycharmProjects/few-shot/dataset/yelp/proc/ppl_all.bin'

        # flows related
        self.anneal_function = 'logistic'
        self.k = 0.0025
        self.x0 = 2000
        self.use_flow = True
        self.separate = False
        self.flow_type = 'realNVP' if self.use_flow else None  # The type of flow to be used, none, scf or glow
        self.flow_nums = 6 if self.use_flow else None  # Number of flows in the NF
        self.flow_units = [self.hidden_size // 2, self.hidden_size // 4,
                           self.hidden_size // 2]  # Size of MLP in scf flows

        # loss factor
        self.nf_factor = 0.01
        self.rec_factor = 1
        self.cyc_factor = 1
        self.adv_factor = 1

        # pretrain
        self.load_pretrain = True
        self.pretrain_epochs = 2

        # train
        self.d_step = 5
        self.warmup_steps = 3000
        self.max_lr = 8e-4
        self.min_lr = 5e-5
        self.init_lr = 1e-5

        self.epochs = 10
        self.max_grad_norm = 5.0

        self.teacher_forcing_ratio = 0.0
        self.attention_dropout = 0.1
        self.dropout = 0.1


class ImdbConfig(Config):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        # model
        self.seed = 16
        self.embedding_size = 512
        self.nlayers = 1
        self.bidirectional = True
        self.hidden_size = 512
        self.max_length = 40
        self.batch_size = 128
        self.num_filters = 64

        # data
        self.dataset = "imdb"
        self.raw_data_dir = './data/imdb/raw'
        self.data_dir = './data/imdb/prepro'
        self.save_dir = "./save"
        self.pretrain_ckpt_dir = './save/imdb/pretrain_ckpt'
        self.classifier_path = '/home/hyh/PycharmProjects/few-shot/classification/imdb/fasttext/fasttext_classifier.bin'
        self.ppl_path = '/home/hyh/PycharmProjects/few-shot/dataset/imdb/proc/ppl_all.bin'

        # flows related
        self.anneal_function = 'logistic'
        self.k = 0.0025
        self.x0 = 2000
        self.use_flow = True
        self.separate = True
        self.flow_type = 'realNVP' if self.use_flow else None  # The type of flow to be used, none, scf or glow
        self.flow_nums = 6 if self.use_flow else None  # Number of flows in the NF
        self.flow_units = [self.hidden_size // 2, self.hidden_size // 4,
                           self.hidden_size // 2]  # Size of MLP in scf flows

        # loss factor
        self.nf_factor = 0.01
        self.rec_factor = 2
        self.rev_factor = 1
        self.cyc_factor = 2
        self.adv_factor = 1

        # pretrain
        self.load_pretrain = True
        self.pretrain_epochs = 4

        # train
        self.d_step = 1
        self.warmup_steps = 3000
        self.max_lr = 8e-4
        self.min_lr = 5e-5
        self.init_lr = 1e-5

        self.epochs = 12
        self.max_grad_norm = 5.0

        self.teacher_forcing_ratio = 0.0
        self.attention_dropout = 0.1
        self.dropout = 0.1


class GYAFCConfig(Config):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        # model
        self.seed = 11
        self.embedding_size = 768
        self.nlayers = 1
        self.bidirectional = True
        self.hidden_size = 768
        self.num_filters = 64
        self.max_length = 35
        self.batch_size = 32

        # data
        self.dataset = "gyafc"
        self.raw_data_dir = './data/gyafc/raw'
        self.data_dir = './data/gyafc/prepro'
        self.save_dir = "./save"
        self.pretrain_ckpt_dir = './save/gyafc/pretrain_ckpt'
        self.classifier_path = '/home/hyh/PycharmProjects/few-shot/classification/gyafc/fasttext/fasttext_classifier.bin'
        self.ppl_path = '/home/hyh/PycharmProjects/few-shot/dataset/gyafc/proc/ppl_all.bin'

        # flows related
        self.anneal_function = 'logistic'
        self.k = 0.0025
        self.x0 = 2000
        self.use_flow = True
        self.separate = False
        self.flow_type = 'realNVP' if self.use_flow else None  # The type of flow to be used, none, scf or glow
        self.flow_nums = 10 if self.use_flow else None  # Number of flows in the NF
        self.flow_units = [self.hidden_size // 2, self.hidden_size // 4,
                           self.hidden_size // 2]  # Size of MLP in scf flows

        # loss factor
        self.nf_factor = 0.01
        self.rec_factor = 1
        self.rev_factor = 1
        self.cyc_factor = 1
        self.adv_factor = 1
        self.dist_factor = 1

        # pretrain
        self.load_pretrain = True
        self.pretrain_epochs = 5

        # train
        self.d_step = 5
        self.warmup_steps = 3000
        self.max_lr = 8e-4
        self.min_lr = 5e-5
        self.init_lr = 1e-5

        self.epochs = 20
        self.max_grad_norm = 5.0

        self.teacher_forcing_ratio = 0.0
        self.attention_dropout = 0.1
        self.dropout = 0.1


############# map ##########
LABEL_MAP = {
    "gyafc": ["0", "1"],
    "yelp": ["0", "1"],
    "imdb": ["0", "1"],
    "debug": ["0", "1"],
}

CONFIG_MAP = {
    "debug": DebugConfig,
    "yelp": YelpConfig,
    "imdb": ImdbConfig,
    "gyafc": GYAFCConfig
}

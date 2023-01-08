import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.models.deeprec.deeprec_utils import cal_metric

class LSTUR_MODEL(object):
    def __init__(self):
        epochs = 1
        seed = 40
        batch_size = 32

        # Options: demo, small, large
        MIND_type = 'small'
        #tmpdir = TemporaryDirectory()
        data_path = '/home/byp/severaltrys/poison/lstur/datas'

        self.train_news_file = os.path.join(data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
        self.valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
        self.wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
        self.userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
        self.wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
        yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')
        mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
        if not os.path.exists(self.train_news_file):
            download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
            
        if not os.path.exists(self.valid_news_file):
            download_deeprec_resources(mind_url, \
                                    os.path.join(data_path, 'valid'), mind_dev_dataset)
        if not os.path.exists(yaml_file):
            download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                                    os.path.join(data_path, 'utils'), mind_utils)

        self.hparams = prepare_hparams(yaml_file, 
                                wordEmb_file=self.wordEmb_file,
                                wordDict_file=self.wordDict_file, 
                                userDict_file=self.userDict_file,
                                batch_size=batch_size,
                                epochs=epochs)
        self.iterator = MINDIterator
        self.seed = seed
        # self.model = LSTURModel(self.hparams, self.iterator, seed=self.seed)
        # self.pre_train_model()

    def pre_train_model(self):
        self.model = LSTURModel(self.hparams, self.iterator, seed=self.seed)
        # tqdm_util = tqdm(
        #     self.model.train_iterator.load_data_from_file(
        #         self.train_news_file, self.train_behaviors_file
        #     )
        # )

        # for batch_data_input in tqdm_util:
        #     x, y = self.model._get_input_label_from_iter(batch_data_input)
        #     print(self.model.out_gradients([x, y, 0]))
            # print(self.model.model.)
        self.model.fit(self.train_news_file, self.train_behaviors_file, self.valid_news_file, self.valid_behaviors_file)

    def run_eval(self, news_file=None, behaviors_file= None):
        mrr = self.model.run_eval(self.valid_news_file, self.valid_behaviors_file)
        return mrr

    def eval_on_batch(self, input_1, input_2, input_3):
        input_4 = input_3.reshape(input_3.shape[0], 1, input_3.shape[1])
        input_feat = [input_1, input_2, input_4]
        pred_rslt = self.model.scorer.predict_on_batch(input_feat)
        return pred_rslt

    def retrain(self, poisoned_news_file):
        '''
            input_feat : ['user_index_batch', clicked_title_batch', candidate_title_batch']
            input_label: ['labels']

        '''
        # self.model = LSTURModel(self.hparams, self.iterator, seed=self.seed)
        self.model.fit(poisoned_news_file, self.train_behaviors_file, self.valid_news_file, self.valid_behaviors_file)
    
    def metric(self, group_labels, group_preds):
        ans = cal_metric(group_labels, group_preds, ['mean_mrr'])
        return ans
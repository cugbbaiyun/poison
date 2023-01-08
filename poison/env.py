import numpy as np
import torch as th
from tqdm import tqdm
import pickle
from lstur_model import LSTUR_MODEL
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import prepare_hparams


class Env(object):
    def __init__(self, params, iterator_creator=None, poisoned_file = None):
        self.poison_model = LSTUR_MODEL()
        self.params = params
        self.poisoned_file = poisoned_file
        self.train_iterator = iterator_creator(params, params.npratio, '\t')
        self.word2vec = self._init_embedding(params.wordEmb_file)
        self.dict = self.load_dict(params.wordDict_file)
        self.index2word = {}
        self.tqdm_util = None
        self.action_space = len(self.dict) + 1
        self.observation_space = [32, 50, 30, 300]
        self.input_feat = None
        self.input_label = None
        self.init_index2word()
        self.pre_train_model()
    
    def pre_train_model(self):
        print("Training poison model:")
        self.poison_model.pre_train_model()
        print("Pre-train end")
    
    def init_index2word(self):
        for k, v in self.dict.items():
            self.index2word[v] = k
    
    def load_dict(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _init_embedding(self, file):
        return np.load(file)

    def prepare_news_generator(self):
        for c in range(1, len(self.train_iterator.index2nid)):
            yield c, self.train_iterator.index2nid[c], self.train_iterator.news_title_index[c], c == len(self.train_iterator.index2nid) - 1
 
    def prerare_iterator(self, news_file, behaviors_file):
        # self.tqdm_util = tqdm(self.train_iterator.load_data_from_file(news_file, behaviors_file))
        self.train_iterator.init_news(news_file)
        self.train_iterator.init_poisoned_news(news_file)
        self.train_iterator.init_behaviors(behaviors_file)
        self.news_generator = self.prepare_news_generator()

    def next_batch(self):
        return next(self.news_generator)
    
    def cal_metric(self, pred):
        preds = []
        preds.extend(np.reshape(pred, -1))
        _, group_labels, group_preds = self.poison_model.model.group_labels(
            self.eval_labels, preds, self.imp_indexes
        )
        ans = self.poison_model.metric(group_labels, group_preds)
        return ans

    def cal_reward(self):
        before_metric= self.poison_model.run_eval()
        mrr1 = before_metric['mean_mrr']
        self.poison_model.retrain(self.poisoned_file)
        after_metric= self.poison_model.run_eval()
        mrr2 = after_metric['mean_mrr']
        print("Retraing: before: {}, after: {}".format(mrr1, mrr2))
        diff = mrr1 - mrr2
        return diff
        # return 0

    def update_news(self, selected_word_index, to_word_index):
        words = []
        for i in range(len(self.current_news_title)):
            index = self.current_news_title[i].item()
            if index == 0:
                continue
            words.append(self.index2word[index])
        if selected_word_index > len(words):
            return -1
        # old_index = words[selected_word_index]
        self.current_news_title[selected_word_index] = to_word_index
        words[selected_word_index] = self.index2word[to_word_index]
        nid = self.current_news_nid
        self.train_iterator.update_poisoned_news(nid, words)
        return 0

    def select_word(self, title):
        valid_len = 0
        for c in title:
            if c == 0:
                break
            valid_len += 1
        return np.random.randint(valid_len) 

    def get_freq(self, word_index):
        if word_index not in self.train_iterator.word_freq.keys():
            word = self.index2word[word_index].lower()
            word_index = self.train_iterator.word_dict[word.lower()] 
        if word_index not in self.train_iterator.word_freq.keys():
            return 1
        return self.train_iterator.word_freq[word_index]
    
    def cal_replace_cost(self, old_word_index, new_word_index):
        old_word_emb = self.word2vec[old_word_index]
        new_word_emb = self.word2vec[new_word_index]
        distance = ((old_word_emb - new_word_emb) ** 2).sum()
        old_word_freq = self.get_freq(old_word_index)
        cost = old_word_freq / distance
        return cost

    def state(self):
        selected_index = self.select_word(self.current_news_title)
        return self.current_news_title, selected_index
        
    def step(self, selected_word_index, action):
        # if self.current_cost >= 2:
        #     done = True 
        #     self.train_iterator.write_poisoned_news(self.poisoned_file)
        #     r = 0
        #     return self.current_news_title, 0, 0, done, None

        done = False
        new_word_index = action.item()
        old_word_index = self.current_news_title[selected_word_index]
        update_success = self.update_news(selected_word_index, new_word_index)
        if update_success < 0:
            self.current_cost += 3.0 
            r = 0.
        else:
            cost = self.cal_replace_cost(old_word_index, new_word_index)
            self.current_cost += cost
            r = self.cal_reward()
            next_title, selected_word_index = self.state()
        if self.current_cost >= 3.0:
            done = True
        if done:
            self.train_iterator.write_poisoned_news(self.poisoned_file)
            print("End of this news")
        return next_title, selected_word_index, r, done, None
    
    # def step(self, selected_word_index, action):
    #     selected_word_index = selected_word_index
    #     new_word_index = action.item()
    #     old_word_index = self.current_news_title[selected_word_index]
    #     old_word = self.update_news(selected_word_index, new_word_index)
    #     self.train_iterator.write_poisoned_news(self.poisoned_file)
    #     cost = self.cal_replace_cost(old_word_index, new_word_index)
    #     self.current_cost += cost
    #     print('Current cost: ', self.current_cost)
    #     r = self.cal_reward()
    #     next_title, selected_word_index, done = self.state()
    #     if done:
    #         print("End of all news")
    #     if self.current_cost > 5.0:
    #         print("Too much cost, end this epsiode")
    #     # if done or self.current_cost > 5.0:
    #     #     done = True
    #     return next_title, selected_word_index, r, done, None

        # # shape = self.input_feat[1].shape
        # his_shape = self.input_feat[3].shape
        # self.input_feat[1] = self.input_feat[1].reshape([self.input_feat[1].shape[0]*self.input_feat[1].shape[1], self.input_feat[1].shape[2], -1])
        # selected_word_index = selected_word_index.reshape([self.input_feat[1].shape[0], -1])
        # self.input_feat[3] = self.input_feat[3].reshape([-1, 1]) #[1600]
        # action = action.reshape([self.input_feat[1].shape[0], -1])
        # for i in range(self.input_feat[1].shape[0]):
        #     loc = selected_word_index[i]
        #     if action[i][loc] == self.action_space - 1:
        #         continue
        #     self.input_feat[1][i][selected_word_index[i]] = action[i][loc].item() + 1
        #     self.update_news(i, self.input_feat[3][i], selected_word_index[i], to_word_index = action[i][loc].item() + 1)
        # self.input_feat[1] = self.input_feat[1].reshape(shape)
        # print(self.poisoned_file)
        # self.train_iterator.write_poisoned_news(self.poisoned_file)

    def reset(self, news_file, behaviors_file):
        # self.pre_train_model()
        self.prerare_iterator(news_file, behaviors_file)
        self.current_cost = 0.0
        return self.next_news()
    
    def next_news(self):
        self.current_cost = 0
        news_index, news_nid, news_title, is_last = self.next_batch() 
        self.current_news_index = news_index
        self.current_news_nid = news_nid
        self.current_news_title = news_title
        self.current_done = is_last
        selected_index = self.select_word(news_title)
        return news_title, selected_index, is_last
    
    # def state(self):
    #     news_index, news_nid, news_title, is_last = self.next_batch()
    #     self.current_news_index = news_index
    #     self.current_news_nid = news_nid
    #     self.current_news_title = news_title
    #     self.current_done = is_last
    #     selected_index = self.select_word(news_title)
    #     return news_title, selected_index, is_last

        # input_feat = [
        #         batch["user_index_batch"],
        #         batch["clicked_title_batch"],
        #         batch["candidate_title_batch"],
        #         batch['his_list'],
        # ]
        # input_label = batch["labels"]
        # self.input_feat = input_feat
        # self.input_label = input_label
        # self.eval_labels = []
        # eval_label= self.input_label[:, 0].reshape(self.input_label.shape[0], 1)
        # self.eval_labels.extend(np.reshape(eval_label, -1))
        # self.imp_indexes = []
        # imp_index = batch['impression_index_batch']
        # self.imp_indexes.extend(np.reshape(imp_index, -1))
        # return input_feat, input_label

    def title2emb(self, title):
        from operator import itemgetter
        emb = np.array(list(itemgetter(*title)(self.word2vec)))
        return emb


def main():
    embedding_file = '/home/byp/severaltrys/poison/lstur/datas/utils/embedding_all.npy'
    dict_file = '/home/byp/severaltrys/poison/lstur/datas/utils/word_dict_all.pkl'
    userdict_file = '/home/byp/severaltrys/poison/lstur/datas/utils/uid2index.pkl'
    train_news_data = '/home/byp/severaltrys/poison/lstur/datas/train/news.tsv'
    train_behaviors_data = '/home/byp/severaltrys/poison/lstur/datas/train/behaviors.tsv'
    yaml_file = '/home/byp/severaltrys/poison/lstur/datas/utils/lstur.yaml'
    poison_news_data = '/home/byp/severaltrys/poison/lstur/datas/train/news_poisoned.tsv'
    from shutil import copyfile
    copyfile(train_news_data, poison_news_data)
    batch_size = 32
    epochs = 5

    params = prepare_hparams(yaml_file,
                            wordEmb_file = embedding_file,
                            wordDict_file = dict_file,
                            userDict_file = userdict_file,
                            batch_size = batch_size,
                            epochs = epochs)
    env = Env(params, MINDIterator, poisoned_file = poison_news_data)
    # print('init finish')
    env.prerare_iterator(train_news_data, train_behaviors_data)
    feat, label = env.state()
    emb = env.title2emb(feat[1])
    print(emb.shape) # [32, 50, 30, 300]
    print(feat[1].shape) #[32, 50, 30]
    can_emb = env.title2emb(feat[2])
    print(can_emb.shape) # [32, 5, 30, 300]
    print(feat[2].shape) # [32, 5, 30]
# main()
if __name__  == '__main__':
    main()
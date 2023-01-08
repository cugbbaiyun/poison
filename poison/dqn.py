import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from env import Env
import numpy as np
from tqdm import tqdm
import pickle
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import prepare_hparams

class dqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(dqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        self.rnn = nn.GRU(input_size = 300, hidden_size = 1024, batch_first = True)
        self.word_mlp = nn.Linear(1024, 300) 
        self.activ1 = nn.ReLU()
        # self.word_out = nn.Linear(512, 300)
        self.fc1 = nn.Linear(self.observation_dim, int(self.observation_dim / 2))
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(int(self.observation_dim / 2), self.action_dim)

    def get_word_embedding(self, title_embeddings):
        word_embed = title_embeddings.reshape(-1, 30, 300)
        out, _ = self.rnn(word_embed)
        out = out.squeeze(dim = 1)
        out2 = self.word_mlp(out)
        out2 = self.activ1(out2)
        out3 = torch.mul(out2, title_embeddings)
        out3 = out3.sum(dim = -1)
        indexs = out3.argmax(dim = -1) 
        selected_word_embed = []
        for i in range(0, title_embeddings.shape[0]):
            selected_word_embed.append(out2[i][indexs[i]]) 
        return torch.stack(selected_word_embed).unsqueeze(dim = 1)

    def select_word(self, word_embeddings):
        # word_embeddings = word_embeddings.reshape(, 300)
        word_embed = word_embeddings.reshape(-1, 30, 300)
        word_embeddings = word_embeddings.reshape(-1, 30, 300)
        # with torch.no_grad():
        out, _ = self.rnn(word_embed)
        out = out.squeeze(dim = 1)
        out2 = self.word_mlp(out)
        out2 = self.activ1(out2)
        out2 = torch.mul(out2, word_embeddings)
        out2 = out2.sum(dim = -1)
        return out2.argmax(dim = -1)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        return x

    def get_action(self, title_emb, epsilon):
        title_emb = title_emb.reshape(-1, 300)
        selected_word_index = self.select_word(title_emb)
        selected_word_index = selected_word_index.reshape(-1).item()
        word_emb = title_emb[selected_word_index].reshape(-1)
        title_emb = title_emb.reshape(-1)
        observation = torch.concat([title_emb, word_emb], dim = -1)
        observation = observation.reshape(-1, observation.shape[-1])
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(dim=-1)[-1]
        else:
            action = torch.randint(low = 0, high = self.action_dim, size = [observation.shape[0]])
        return action, selected_word_index 

class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    # def store(self, title_emb, word_emb, action, reward, next_title_emb, next_word_emb, done):
    def store(self, title_emb, action, reward, next_title_emb, done):
        title_emb = np.expand_dims(title_emb, 0)
        # word_emb = np.expand_dims(word_emb, 0)
        next_title_emb = np.expand_dims(next_title_emb, 0)
        # next_word_emb = np.expand_dims(next_word_emb, 0)
        # self.memory.append([title_emb, word_emb, action, reward, next_title_emb, next_word_emb, done])
        self.memory.append([title_emb, action, reward, next_title_emb, done])

    def sample(self, size):
        batch = random.sample(self.memory, size)
        # title_emb, word_emb, action, reward, next_title_emb, next_word_emb, done = zip(* batch)
        title_emb, action, reward, next_title_emb, done = zip(* batch)
        title_emb = np.concatenate(title_emb, axis = 0)
        # word_emb = np.concatenate(word_emb, axis = 0)
        next_title_emb = np.concatenate(next_title_emb, axis = 0)
        # next_word_emb = np.concatenate(next_word_emb, axis = 0)
        # return title_emb, word_emb, action, reward, next_title_emb, next_word_emb, done
        return title_emb, action, reward, next_title_emb, done

    def __len__(self):
        return len(self.memory)


def training(buffer, batch_size, model, optimizer, gamma, loss_fn):
    # title_emb, word_emb, action, reward, next_title_emb, next_word_emb, done = buffer.sample(batch_size)
    title_emb, action, reward, next_title_emb, done = buffer.sample(batch_size)

    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    title_emb = torch.FloatTensor(title_emb)
    word_emb = model.get_word_embedding(title_emb)
    # selected_word_index = model.select_word(title_emb)
    # word_emb = torch.FloatTensor(word_emb).unsqueeze(dim = 1)
    next_title_emb = torch.FloatTensor(next_title_emb)
    next_word_emb = model.get_word_embedding(next_title_emb)
    # next_selected_word_index = model.select_word(next_title_emb)
    # next_word_emb = torch.FloatTensor(next_word_emb).unsqueeze(dim = 1)
 
    # observation = torch.concat([title_emb, word_emb], dim = 1)
    observation = torch.concat([title_emb, word_emb], dim = 1)
    observation = observation.reshape([observation.shape[0], -1])
    # next_observation = torch.concat([next_title_emb, next_word_emb], dim = 1)
    next_observation = torch.concat([next_title_emb, next_word_emb], dim = 1)
    next_observation = next_observation.reshape([next_observation.shape[0], -1])

    q_values = model.forward(observation)
    next_q_values = model.forward(next_observation)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0].detach()
    expected_q_value = reward + next_q_value * (1 - done) * gamma

    loss = loss_fn(q_value, expected_q_value.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    embedding_file = '/home/byp/severaltrys/poison/lstur/datas/utils/embedding_all.npy'
    dict_file = '/home/byp/severaltrys/poison/lstur/datas/utils/word_dict_all.pkl'
    userdict_file = '/home/byp/severaltrys/poison/lstur/datas/utils/uid2index.pkl'
    train_news_data = '/home/byp/severaltrys/poison/lstur/datas/train/news.tsv'
    train_behaviors_data = '/home/byp/severaltrys/poison/lstur/datas/train/behaviors.tsv'
    yaml_file = '/home/byp/severaltrys/poison/lstur/datas/utils/lstur.yaml'
    poison_news_data = '/home/byp/severaltrys/poison/lstur/datas/train/news_poisoned.tsv'
    batch_size = 32
    epochs = 5

    params = prepare_hparams(yaml_file,
                            wordEmb_file = embedding_file,
                            wordDict_file = dict_file,
                            userDict_file = userdict_file,
                            batch_size = batch_size,
                            epochs = epochs)
    env = Env(params, MINDIterator, poisoned_file= poison_news_data)
    env.prerare_iterator(train_news_data, train_behaviors_data)

    epsilon_init = 0.9
    epsilon_min = 0.01
    decay = 0.995
    capacity = 500
    exploration = 100
    batch_size = 50
    episode = 1000
    learning_rate = 1e-3
    gamma = 0.99
    loss_fn = nn.MSELoss()

    action_dim = env.action_space
    history_length = env.observation_space[-3]
    title_length = env.observation_space[-2]
    observation_dim = (30 + 1) * 300

    model = dqn(observation_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    buffer = replay_buffer(capacity)
    epsilon = epsilon_init
    weight_reward = None

    feat, _ , done = env.reset(train_news_data, train_behaviors_data)
    # feat, selected_word_index, done = env.reset(train_news_data, train_behaviors_data)
    # feat, selected_word_index, done = env.next_news(train_news_data, train_behaviors_data)
    title_emb = env.title2emb(feat)
    # word_emb = env.word2vec[feat[selected_word_index]]

    for i in range(episode):
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0
        # selected_word_index = select_word(feat)
        # action = model.get_action(torch.FloatTensor(word_emb), torch.FloatTensor(title_emb), epsilon)
        action, selected_word_index = model.get_action(torch.FloatTensor(title_emb), epsilon)
        train_flag = False
        next_feat, _, reward, done, info = env.step(selected_word_index, action)
        # next_feat, next_word_index, reward, done, info = env.step(selected_word_index, action)
        next_title_emb = env.title2emb(next_feat)
        # next_word_emb = env.word2vec[feat[selected_word_index]]
        # buffer.store(title_emb, word_emb, action.item(), reward, next_title_emb, next_word_emb, done)
        buffer.store(title_emb, action.item(), reward, next_title_emb, done)
        reward_total += reward
        feat = next_feat
        if len(buffer) % exploration == 0:
            print('<<<<<<<<<<< training dqn: <<<<<<<<<')
            training(buffer, batch_size, model, optimizer, gamma, loss_fn)
            train_flag = True
        if done:
            if not weight_reward:
                weight_reward = reward_total
            else:
                weight_reward = 0.99 * weight_reward + 0.01 * reward_total
            print('episode: {}  reward: {}  epsilon: {:.2f}  train:  {}  weight_reward: {:.3f}'.format(i+1, reward_total, epsilon, train_flag, weight_reward))
            # feat, selected_word_index, done = env.reset(train_news_data, train_behaviors_data)
            # feat, selected_word_index, done = env.next_news(train_news_data, train_behaviors_data)
            # feat, selected_word_index, done = env.next_news()
            feat, _, done = env.next_news()
            if done:
                break
            title_emb = env.title2emb(feat)
            # word_emb = env.word2vec[feat[selected_word_index]]
            # break
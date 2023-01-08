from tkinter import HIDDEN
import numpy as np
import torch as th
import torch.nn as nn

class TitleEmbedding(object):
    def __init__(self, *args):
        self.args = args
        self.state_dim = 300
        self.hidden_dim = 128
        self.out_dim = 32 
        self.title_length = 30
        self.rnn = nn.GRU(input_size = self.state_dim, hidden_size = self.hidden_dim, batch_first = True, num_layers = 2)
        self.hidden_dim2 = 64 
        self.out_dim2 = 16
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.out_dim),
            nn.ReLU()
        )
    
    def forward(self, title_embedding):
        '''
            title_embedding: [batch, history_length, title_length, feature_size] [32, 50, 30, 300]
        '''
        original_shape = None
        if len(title_embedding.shape) == 4:
            # [batch, histories, title_length, state_dim] -> [batch * histories, title_length, state_dim]
            original_shape = title_embedding.shape
            title_embedding = title_embedding.reshape([-1, title_embedding.shape[-2], title_embedding.shape[-1]])
        self.hidden_state = th.zeros([2, title_embedding.shape[0], self.hidden_dim])
        # [batch * histories, title_length, state_dim] 
        X, self.hidden_state = self.rnn(title_embedding, self.hidden_state) #[]
        # [batch * histories, hidden_dim]
        X = X[:, -1, :]
        # [batch * histories, hidden_dim] -> [batch * histories, output_dim]
        X = self.encoder(X)
        # [batch, histories, output_dim]
        if original_shape:
            X = X.reshape([original_shape[0], original_shape[1], -1])
        return X

def main():
    input = th.ones([32,50,30, 300])
    title_embedding = TitleEmbedding()
    ret = title_embedding.forward(input)
    print(ret.shape)
main()
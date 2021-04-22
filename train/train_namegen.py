from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset
from dotmap import DotMap
from typing import Dict

import collections
import math

import numpy as np
import torch
from typing import Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NameGen3(pl.LightningModule):
    def __init__(self,  
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        label_embedder: Optional[TokenEmbedder] = None ,
        contextualizer: Optional[Seq2SeqEncoder]=None,
        hidden_size: int = 100,
        dropout:float = 0.2,
        num_layers = 2):
        super().__init__()

        self.hidden_size = hidden_size
        self.text_field_embedder = text_field_embedder
        self.label_embedder = label_embedder
        self.contextualizer = contextualizer
        self.num_layers = num_layers
        
        self.output_size = vocab.get_vocab_size('tokens')
        self.hidden_size = hidden_size

        ### Parameters ###
        self.lstm = nn.LSTM( input_size=text_field_embedder.get_output_dim() + label_embedder.get_output_dim() , hidden_size=hidden_size, 
                num_layers=self.num_layers,batch_first=True)
        
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def init_weights(self):
        
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
                
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))


    def forward(self, tokens, culture, hidden_state=None):
        car_embeddings = self.text_field_embedder(tokens)
        cat_embeds = self.label_embedder(culture)
        

        output, hidden_state = self.lstm(car_embeddings, hidden_state)

        output = self.decoder(output)

        return output, hidden_state

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        return h,c

    def training_step(self, batch, batch_idx):
        src, tgt, lengths = batch
        hidden_state = self.init_hidden(src.shape[0])
        


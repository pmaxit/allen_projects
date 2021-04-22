from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import SoftmaxLoss
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Perplexity
from allennlp.nn.initializers import InitializerApplicator
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from typing import *
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@Model.register('next_token_model2')
class NameGenModel2(Model):
    def __init__(self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        label_embedder: TokenEmbedder,
        contextualizer: Optional[Seq2SeqEncoder]=None,
        hidden_size: int = 100,
        dropout:float = 0.2,
        num_layers = 2):

        super().__init__(vocab)
        self.hidden_size = hidden_size
        self.text_field_embedder = text_field_embedder
        self.label_embedder = label_embedder
        self.contextualizer = contextualizer
        self.num_layers = num_layers
        
        self.output_size = vocab.get_vocab_size('tokens')
        self.hidden_size = hidden_size

        ### Parameters ###
        self.lstm = nn.LSTM( input_size=text_field_embedder.get_output_dim() + label_embedder.get_output_dim() ,
                 hidden_size=hidden_size, num_layers=self.num_layers,
                 batch_first=True, dropout=0.3)
        
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.bn = torch.nn.BatchNorm1d(text_field_embedder.get_output_dim() + label_embedder.get_output_dim())
        self.fc3 = nn.Linear(self.output_size, self.output_size)

        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

        ### Loss Criterion ###
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        #self.init_weights()


    def _compute_loss(self, next_token_prob, forward_targets):
        mask = forward_targets > 2
        non_masked_targets = forward_targets.masked_select(mask)

        non_masked_next_token = next_token_prob.masked_select(mask.unsqueeze(-1)).view(-1, self.output_size)

        return self.criterion(non_masked_next_token, non_masked_targets)

    def forward(self, inputs: TextFieldTensors, 
                targets: TextFieldTensors ,
                culture: torch.Tensor,
                hidden:Tuple[torch.Tensor, torch.Tensor]=None):
        token_ids = inputs['characters']['tokens']
        forward_targets = targets['characters']['tokens']
        batch_size, seq_length = token_ids.shape

        if hidden is None:
            self.init_hidden(batch_size=batch_size)

        car_embeddings = self.text_field_embedder(inputs)
        cat_embeddings = self.label_embedder(culture)

        # combined inputs
        cat_embeds = cat_embeddings.unsqueeze(1).repeat(1,seq_length, 1)

        combined_input = torch.cat([car_embeddings, cat_embeds], 2)
        # calculate lengths

        lengths = (token_ids > 0).sum(axis=1)
        combined_input = nn.utils.rnn.pack_padded_sequence(combined_input, lengths, batch_first=True, enforce_sorted=False )
        # pass through LSTM
        lstm_out, hidden = self.lstm(combined_input, hidden)

        # get it back
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_length)

        # output
        # get output for every location
        output = self.fc2(self.dropout(lstm_out))

        loss = self._compute_loss(output, forward_targets)

        # find loss

        return {'loss': loss ,
                'next_token_prob': output[:,-1,:],
                'hidden': (hidden[0].detach(), hidden[1].detach())}

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
                
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        return (h,c)
    

    
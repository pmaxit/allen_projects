
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import SoftmaxLoss
from allennlp.modules.text_field_embedders import TextFieldEmbedder
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

@Model.register('next_token_model')
class NameGenModel2(Model):
    def __init__(self, 
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder,
            dropout: float = 0.2,
            num_samples: int = 5,
            hidden_size: int = 16,
            num_layers: int = 2,
            initializer: InitializerApplicator=None,
            scheduled_sampling_ratio:float = 0.2,
            max_tokens: int=10,
            **kwargs):
    
        super(NameGenModel2, self).__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        #self._contextualizer = contextualizer
        if True:
            self._contextualizer = nn.LSTM(input_size = 50,
                 hidden_size= self._hidden_size, num_layers= self._num_layers, batch_first=True)
        self._num_labels = self.vocab.get_vocab_size('character_vocab')

        self.decoder = nn.Linear(16,  self._num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
        self._perplexity = Perplexity()
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._start_index = self.vocab.get_token_index(START_SYMBOL, "character_vocab")
        self._end_index = self.vocab.get_token_index(END_SYMBOL, 'character_vocab')

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)

        self.init_weights()

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
        h = torch.zeros(self._num_layers, batch_size, self._hidden_size).to(device)
        c = torch.zeros(self._num_layers, batch_size, self._hidden_size).to(device)

        return (h,c)

    def _compute_loss(self, 
            lm_embeddings: torch.Tensor,
            forward_targets: torch.Tensor,
            backward_targets: torch.Tensor = None):
        # get mask
        mask = forward_targets > 0
        non_masked_targets = forward_targets.masked_select(mask)

        non_masked_embeddings = lm_embeddings.masked_select(mask.unsqueeze(-1)).view(-1, self._num_labels)

        # calculate softmax loss
        return self.criterion(non_masked_embeddings, non_masked_targets)

        #return self._softmax_loss(non_masked_embeddings, non_masked_targets)

    def forward(self, tokens: TextFieldTensors, hidden_state:Tuple[torch.tensor, torch.tensor]=None) -> Dict[str, torch.tensor]:
        """ Computes the averaged forward & backward, if language model is bidirectional """
        token_ids = tokens['characters']['tokens']
        # initialize hidden state
        if hidden_state is None:
            hidden_state = self.init_hidden(token_ids.shape[0])

        mask = get_text_field_mask(tokens)
        embeddings = self._text_field_embedder(tokens)
        contextual_embeddings, hidden_state = self._contextualizer(embeddings, hidden_state)
        
        contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)
        
        #[batch_size, seq_len, vocab_len]
        output = self.decoder(contextual_embeddings_with_dropout)


        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:,0:-1] = token_ids[:,1:]
        
        forward_loss = self._compute_loss(output, forward_targets)

        num_targets = (torch.sum((forward_targets > 0).long()))
        average_loss = forward_loss / num_targets.float()

        return_dict= {}
        self._perplexity(average_loss)

        if num_targets > 0:
            return_dict.update(
                {
                    'loss': average_loss,
                    'next_token_logit': output[:,-1,:],
                    'hidden_state': hidden_state
                }
            )
        
        return return_dict
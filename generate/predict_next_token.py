from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional
import logging

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor
import torch
import torch.nn.functional as F
from allennlp.data.batch import Batch
from allennlp.data.tokenizers import Token
import sys

logger = logging.getLogger(__name__)

@Predictor.register("next_token_predictor")
class NamePredictor(Predictor):
    def __init__(self, *args, temperature=0.3, **kwargs):
        self.temperature = temperature
        super().__init__(*args, **kwargs)

    def predict_json(self, inputs: JsonDict)-> JsonDict:
        culture = inputs['culture']
        tokens = self._dataset_reader._tokenizer.tokenize(inputs['name'])
        instance = self._dataset_reader.text_to_instance(culture=culture, tokens=tokens)
        return self.predict_instance(instance)


    def predict_instance(self, instance : Instance)->JsonDict:
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('tokens')

        predict_len = 10
        hidden_state = self._model.init_hidden(batch_size = 1, device=torch.device('cpu'))

        predicted_str = ''.join([t.text for t in instance.fields['inputs'].tokens[1:-1]])
        for _ in range(predict_len):
            self._dataset_reader.apply_token_indexers(instance)

            dataset = Batch([instance])
            dataset.index_instances(self._model.vocab)

            outputs = self._model.forward(hidden=hidden_state, **dataset.as_tensor_dict())

            # get the logits
            logits = F.log_softmax(outputs['next_token_prob'],dim=-1)
            hidden_state = outputs['hidden']
            
            output_dist = logits.data.view(-1).div(self.temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]

            sos_idx = self._model.vocab.get_token_index('SOS')
            eos_idx = self._model.vocab.get_token_index('EOS')
            if top_char in [0, 1, sos_idx, eos_idx]:
                break
            
            top_char = top_char.item()
            # convert back to string

            predicted_char = label_vocab[top_char]
            predicted_str += predicted_char

            # modify the instance
            # create new instance
            #current_tokens = [t for t in instance.fields['tokens'].tokens]
            #current_tokens.append(Token(predicted_char))

            # new instance with only predicted token
            instance = Instance({'inputs': TextField([Token(predicted_char)]),
                    'targets' : TextField([Token(predicted_char)]),
                    'culture': instance.fields['culture']})
        
        return sanitize({'prediction': predicted_str})
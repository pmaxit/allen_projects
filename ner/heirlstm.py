import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules import ConditionalRandomField

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure

from typing import Optional, Dict, Any, List
import torch
from overrides import overrides

@Model.register('heir_lstm')
class HierarchicalLstm(Model):
    def __init__(self,
            vocab: Vocabulary,
            word_embedder: TextFieldEmbedder,
            character_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            character_encoder: Seq2VecEncoder,
            use_crf: bool = False)->None:

            super().__init__(vocab)
            self.use_crf = use_crf
            self._word_embedder = word_embedder
            self._character_embedder = character_embedder
            self._character_encoder = character_encoder
            self._encoder = encoder

            self._classifier = torch.nn.Linear(
                in_features = encoder.get_output_dim(),
                out_features = vocab.get_vocab_size('labels')
            )
            self._crf = ConditionalRandomField(
                vocab.get_vocab_size('labels')
            )
            self._f1 = SpanBasedF1Measure(vocab, 'labels')

    def get_metrics(self, reset:bool = True):
        return self._f1.get_metric(reset)

    @overrides
    def make_output_human_readable(self, output_dict):

        return output_dict

    def _broadcast_tags(self,
                        viterbi_tags: List[List[int]],
                        logits: torch.Tensor) -> torch.Tensor:
        output = logits * 0.
        for i, sequence in enumerate(viterbi_tags):
            for j, tag in enumerate(sequence):
                output[i, j, tag] = 1.
        return output


    def forward(self, tokens, label):
        # split the namespace into characters and tokens since they are not the same shape
        characters = {'characters': tokens['characters']}
        tokens = {'tokens': tokens['tokens']}

        # get the token mask
        mask = get_text_field_mask(tokens)

        # get the character mask
        characters_mask = get_text_field_mask(characters, num_wrapping_dims=1)

        batch_size, sequence_length, word_length = characters_mask.shape

        # embed the character
        embedded_characters =  self._character_embedder(characters)

        # conver the embeddings from 4d embeddings to 3d tensor
        # each word in its own instance a batch
        embedded_characters = embedded_characters.view(batch_size * sequence_length , word_length, -1)
        characters_mask = characters_mask.view(batch_size * sequence_length, word_length)

        # run the character LSTM
        encoded_characters = self._character_encoder(embedded_characters, characters_mask)

        # reshape so that we can concatenate with word
        encoded_characters = encoded_characters.view(batch_size, sequence_length, -1)
        
        #  run the standard NER pipeline
        embedded = self._word_embedder(tokens)
        
        embedded = torch.cat([embedded, encoded_characters], dim=2)
        encoded = self._encoder(embedded, mask)

        classified = self._classifier(encoded)

        if self.use_crf:
            viterbi_tags = self._crf.viterbi_tags(classified, mask)
            viterbi_tags = [path for path, score in viterbi_tags]
            
            broadcasted = self._broadcast_tags(viterbi_tags, classified)

            log_likelihood = self._crf(classified, label, mask)
            self._f1(broadcasted, label, mask)
            output = {}
            output['loss'] = - log_likelihood

        else:
            output = {}
        
            output['logits'] = classified
        if label is not None:
            self._f1(classified, label, mask)

            output['loss'] = sequence_cross_entropy_with_logits(classified, label, mask)

        return output

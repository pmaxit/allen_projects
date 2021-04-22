from typing import Dict

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure
from typing import *
import torch.nn as nn
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


@Model.register("ner_lstm")
class NERLstm(Model):
    def __init__(self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder, 
        encoder: Seq2SeqEncoder)->None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(in_features = encoder.get_output_dim(), 
                   out_features = vocab.get_vocab_size('labels'))
        self._f1 = SpanBasedF1Measure(vocab, 'labels')


    def forward(self,
        tokens: Dict[str, torch.tensor],
        label: Optional[torch.tensor] = None) -> Dict[str, torch.tensor]:

        mask = get_text_field_mask(tokens)
        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)

        classified = self._classifier(encoded)

        self._f1(classified, label, mask)

        output: Dict[str, torch.tensor] = {}

        output['logits'] = classified
        if label is not None:
            output['loss'] = sequence_cross_entropy_with_logits(classified, label, mask)

        return output

    def get_metrics(self, reset:bool = False)->Dict[str, float]:
        return self._f1.get_metric(reset)
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
from allennlp.nn.initializers import InitializerApplicator
from allennlp.data import TextFieldTensors, Vocabulary


from typing import Optional, Dict, Any, List
import torch
from overrides import overrides

class SeqDecoder:
    pass

class NameGenModel(Model):
    def __init__(self, vocab, 
                    source_text_embedder: TextFieldEmbedder,
                    encoder: Seq2SeqEncoder,
                    decoder: SeqDecoder,
                    tied_source_embedder_key: Optional[str]=None,
                    initializer: InitializerApplicator = InitializerApplicator(),
                    **kwargs):
        super().__init__(vocab, **kwargs)

        self._source_text_embedder = source_text_embedder
        self._encoder = encoder
        self._decoder = decoder

        if self._encoder.get_output_dim() != self._decoder.get_output_dim():
            raise ConfigurationError(
                f"Encoder output dimension {self._encoder.get_output_dim()} should be"
                f"equal to decoder dimension {self._decoder.get_output_dim()}."
            )
        if tied_source_embedder_key:
            # A bit of ugly hack to tie embeddings
            # works only for basictextfield embedder
            self._source_text_embedder._token_embedders[
                tied_source_embedder_key
            ] = self._decoder.target_embedder

    @overrides
    def forward(self, 
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None)-> Dict[str, torch.tensor]:
        """ Make forward pass on the encoder and decoder for producing the entire 
        target sequence """

        state = self._encode(source_tokens)
        return self._decoder(state, target_tokens)
    
    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    )-> Dict[str, torch.Tensor]:
        return self._decoder.post_process(output_dict)

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Make forward pass on the encoder """
        embedded_input = self._source_text_embedder(source_tokens)

        source_mask = util.get_text_field_mask(source_tokens)

        encoder_outputs = self._encoder(embedded_input, source_mask)

        return {'source_mask': source_mask, 'encoder_outputs': encoder_outputs}

    @overrides
    def get_metrics(self, reset:bool = False)-> Dict[str, float]:
        return self._decoder.get_metrics(reset)
    

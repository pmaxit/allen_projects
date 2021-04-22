import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import torch

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.util import get_train_and_validation_metrics
from allennlp.data import TensorDict


logger = logging.getLogger(__name__)

@TrainerCallback.register('model_logger')
class ModelLoggerCallback(TrainerCallback):
    def __init__(self,
            serialization_dir: str,
            should_log_inputs: bool=False):

        super().__init__(serialization_dir)
        self._should_log_inputs = should_log_inputs

    def create_instance(self, inputs):
        
        instances= []
        for label, chars in inputs.items():
            fields = {}
            fields['tokens'] = TextField([Token(c) for c in chars ])
            fields['culture'] = LabelField(label)
            instances.append(Instance(fields))
        
        return instances

    def predict_for_instance(self, trainer, inst):
        # first apply token indexers
        # index with vocab
        # model execution

        # return string
        pass
    def on_batch(self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool =True,
        **kwargs)->None:

        if not is_primary:
            return None
        
        # if we only want to do this for the first batch in the first epoch
        if batch_number == 1 and epoch == 0 and self._should_log_inputs:
            logger.info("batch inputs::: PUNEET  :::: ")
            for b, batch in enumerate(batch_inputs):
                self._log_fields(batch, log_prefix='batch_input')

    def _log_fields(self, fields: Dict, log_prefix: str = ""):
        for key, val in fields.items():
            key = log_prefix + "/" + key
            if isinstance(val, dict):
                self._log_fields(val, key)
            elif isinstance(val, torch.Tensor):
                torch.set_printoptions(threshold=2)
                logger.info("%s (Shape: %s)\n%s", key, " x ".join([str(x) for x in val.shape]), val)
                torch.set_printoptions(threshold=1000)
            elif isinstance(val, List):
                logger.info('Field : "%s" : (Length %d of type "%s")', key, len(val), type(val[0]))
            elif isinstance(val, str):
                logger.info('Field : "{}" : "{:20.20} ..."'.format(key, val))
            else:
                logger.info('Field : "%s" : %s', key, val)
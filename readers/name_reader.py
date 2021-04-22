from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer,TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, WhitespaceTokenizer, CharacterTokenizer

import itertools
from typing import Dict, List, cast
from allennlp.common.util import ensure_list
import logging
from allennlp.data import Vocabulary


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@DatasetReader.register('name_dataset')
class NameReader(DatasetReader):
    """ Reads the names from the dataset
        # parameters
        tokenizr: 'Tokenizer, optional (default=whitespacetokenizer)
        token_indexers: 
        max_tokens: if you don't handle truncation at the tokenizer level, you can specify 
        max_tokens here
    """
    def __init__(self, tokenizer: Tokenizer = None,
                token_indexers: Dict[str,TokenIndexer] = None,
                max_tokens: int = None,
                **kwargs) -> None:
            super().__init__(**kwargs)
            self._tokenizer = tokenizer or CharacterTokenizer()
            self._token_indexers = token_indexers or {'characters': SingleIdTokenIndexer(namespace='character_vocab') }
            self._max_tokens = max_tokens

    @overrides
    def _read(self, file_path: str,skip_header: bool=True):
        # generate random names
        # yield instance
        with open(file_path, 'r') as name_file:
            for name in name_file:
                tokens = self._tokenizer.tokenize(name[:-1])
                if skip_header:
                    skip_header=False
                    continue
                yield self.text_to_instance(name, tokens)

    @overrides
    def text_to_instance(self,
        name: str = None,
        tokens: List[Token] = None
        ):
        
        tokens = cast(List[Token], tokens)

        # set max_tokens
        if self._max_tokens is not None:
            tokens = tokens[- self._max_tokens:]
        
        input_field = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {'tokens': input_field}

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance)-> None:
        instance['tokens'].token_indexers = self._token_indexers

def test():
    reader = NameReader()
    instances = reader.read('./data/first_names.all.txt')
    instances = ensure_list(instances)

    # expected few names
    fields = instances[0].fields
    logger.info(fields)
    tokens = [t.text for t in fields['tokens']]
    logger.info(tokens)

    fields = instances[1].fields
    tokens = [t.text for t in fields['tokens']]
    logger.info(tokens)

    instances[0].fields

    # Now we need to create a small vocabulary from our sentence- Note that we have used
    # only character indexers, we we call Vocabulary.from_instances, this will create
    # vocabulary which correspond to the namespaces of each token indexer in our Text Field's

    # build vocabulary
    
    vocab = Vocabulary.from_instances(instances)

    print("This is the token ids vocabulary we created \n")
    print(vocab.get_index_to_token_vocabulary('character_vocab'))

    for instance in instances:
        instance.index_fields(vocab)
    
    # get the tensor dict
    logger.info(instances[0].as_tensor_dict())

if __name__ == '__main__':
    test()

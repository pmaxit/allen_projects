from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer,TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, WhitespaceTokenizer, CharacterTokenizer

import itertools
from typing import Dict, List, cast
from allennlp.common.util import ensure_list
import logging
from allennlp.data import Vocabulary
import glob
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@DatasetReader.register('name_cultgen')
class NameReader(DatasetReader):
    """ Reads the names from dataset"""

    def __init__(self, tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer]= None,**kwargs):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or CharacterTokenizer(lowercase_characters=True, 
                            start_tokens= ['SOS'], end_tokens= ['EOS'])
        self._token_indexers = token_indexers or {'characters': SingleIdTokenIndexer()}


    def read_file(self, file_path: str, skip_header:bool = True):
        with open(file_path, 'r') as name_file:
            for name in name_file:
                tokens = self._tokenizer.tokenize(name[:-1])
                if skip_header:
                    skip_header = False
                    continue
                yield self.text_to_instance(file_path, tokens)
    @overrides
    def _read(self, dir_path: str, skip_header: bool=True):
        for name in glob.glob('./data/names/*.txt'):
            print('reading name ', name)
            yield from self.read_file(name, skip_header=False)
            
    @overrides
    def text_to_instance(self,
        culture:str = None,
        tokens: List[Token]  = None):

        tokens = cast(List[Token], tokens)

        input_field = TextField(tokens[:-1], self._token_indexers)
        output_field = TextField(tokens[1:],self._token_indexers)
        culture = Path(culture).stem
        culture_field = LabelField(culture)

        fields = {'inputs': input_field, 'culture': culture_field, 'targets': output_field}
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance):
        instance['inputs'].token_indexers = self._token_indexers
        instance['targets'].token_indexers = self._token_indexers

def test():
    reader = NameReader()
    instances = reader.read('./data/names')
    instances = ensure_list(instances)
    print(instances[0])
    logger.info("Total names {}".format(len(instances)))
        # expected few names
    fields = instances[0].fields
    logger.info(fields)
    tokens1 = [t.text for t in fields['tokens']]

    logger.info(tokens1)
    logger.info(fields['culture'])

        # build vocabulary
    vocab = Vocabulary.from_instances(instances)

    print("This is the token ids vocabulary we created \n")
    print(vocab.get_index_to_token_vocabulary('tokens'))
    print(vocab.get_index_to_token_vocabulary('labels'))

    for instance in instances:
        instance.index_fields(vocab)

if __name__ == '__main__':
    test()
import logging
from typing import Dict, List, Set
from overrides import overrides
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from ucca_span_parser.tokenizers.multilingual_token import MultilingualToken

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("language")
class LanguageIndexer(TokenIndexer[int]):
    # TODO: Think how to use define the LanguageIndexer superclass. This is because the TokenIndexer class methods
    #  expects a Token class as input and i expect a MultilingualToken, which make it incompatible with the
    #  TokenIndexer class.
    """
    This :class:`LanguageIndexer` represents tokens by their language, as determined
    by the ``lang`` field on ``MultilingualToken``.

    Parameters
    ----------
    namespace : ``str``, optional (default=``languages``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """

    # pylint: disable=no-self-use
    def __init__(self, namespace: str = 'languages',
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self._logged_errors: Set[str] = set()

    @overrides
    def count_vocab_items(self, token: MultilingualToken, counter: Dict[str, Dict[str, int]]):
        lang = token.lang
        if not lang:
            if token.text not in self._logged_errors:
                logger.warning("Token had no language: %s", token.text)
                self._logged_errors.add(token.text)
            lang = 'NONE'
        counter[self.namespace][lang] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens,
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        langs = [token.lang or 'NONE' for token in tokens]

        return {index_name: [vocabulary.get_token_index(lang, self.namespace) for lang in langs]}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in tokens.items()}

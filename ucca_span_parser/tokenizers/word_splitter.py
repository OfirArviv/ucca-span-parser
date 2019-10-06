from typing import List
from overrides import overrides
from allennlp.data import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter, SpacyWordSplitter
from .multilingual_token import MultilingualToken


@WordSplitter.register('spacy-whitespace-multilingual')
class SpacyMultilingualWhitespaceWordSplitter(WordSplitter):

    @overrides
    def split_words(self, sentence: str, lang: str = "en") -> List[Token]:
        if lang == "en":
            spacy_model = "en_core_web_sm"
        elif lang == "de":
            spacy_model = "de_core_news_sm"
        elif lang == "fr":
            spacy_model = "fr_core_news_sm"
        else:
            raise Exception(f'Language {lang} is not implemented in SpacyMultilingualWhitespaceWordSplitter.')

        spacy_tokens: List[Token] = SpacyWordSplitter(language=spacy_model, pos_tags=True, parse=True, ner=True,
                                                      split_on_spaces=True).split_words(sentence)
        return [MultilingualToken(token=token, lang=lang) for token in spacy_tokens]

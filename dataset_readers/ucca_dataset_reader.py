import copy
import os
from random import shuffle
from typing import Dict, List, Iterator, Callable, Tuple
from allennlp.data import DatasetReader, TokenIndexer, Instance, Field
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, MetadataField, LabelField, ArrayField, SpanField, ListField, \
    SequenceLabelField
from allennlp.data.tokenizers.word_splitter import WordSplitter
from overrides import overrides
from ucca.convert import file2passage
from ucca.core import Passage
from htl_suda_ucca_parser import UCCA2tree
from htl_suda_ucca_parser.convert import gerenate_remote
import numpy as np


@DatasetReader.register('ucca-span')
# TODO: Go over
class UccaSpanParserDatasetReader(DatasetReader):
    def __init__(self,
                 word_tokenizer: WordSplitter,
                 token_indexers: Dict[str, TokenIndexer],
                 filter_func: Callable[[List[str]], bool] = None,
                 shuffle: bool = True,
                 lazy: bool = False
                 ) -> None:
        super().__init__(lazy=lazy)
        self.word_tokenizer = word_tokenizer
        self.token_indexers = token_indexers
        self.filter_func = filter_func
        self.shuffle = shuffle

    @staticmethod
    def _get_file_list(dataset_dir: str, dataset_title_prefix: str = "") -> Iterator[Tuple[str, str]]:
        dir_name = os.path.basename(dataset_dir)
        dataset_title = f'{dataset_title_prefix}{dir_name}'
        for file in sorted(os.listdir(dataset_dir)):
            file_path = os.path.join(dataset_dir, file)
            if os.path.isdir(file_path):
                yield from UccaSpanParserDatasetReader._get_file_list(file_path, f'{dataset_title}_')
            else:
                yield (file_path, dataset_title)

    @overrides
    def text_to_instance(self, tokenized_text: [str], passage: Passage, dataset_label: str, lang: str, id: str = None,
                         gold_tree: Passage = None) -> Instance:
        fields: Dict[str, Field] = {}

        word_tokens = self.word_tokenizer.split_words(" ".join(tokenized_text), lang)
        assert len(tokenized_text) == len(word_tokens)
        for i in range(len(word_tokens)):
            assert word_tokens[i].text == tokenized_text[i]
        sentence_field = TextField(word_tokens, self.token_indexers)
        fields["tokens"] = sentence_field

        lang_field = MetadataField(lang)
        fields["lang"] = lang_field

        passage_field = MetadataField(passage)
        fields["passage"] = passage_field

        dataset_label_field = MetadataField(dataset_label)
        fields["dataset_label"] = dataset_label_field

        if id is not None:
            id_field = MetadataField(id)
            fields["id"] = id_field

        if gold_tree is not None:
            gold_ucca_tree_field = MetadataField(gold_tree)
            fields["gold_ucca_tree"] = gold_ucca_tree_field

            gold_primary_tree = UCCA2tree(copy.deepcopy(gold_tree)).convert()
            gold_primary_tree_field = MetadataField(gold_primary_tree)
            fields["gold_primary_tree"] = gold_primary_tree_field

            spans, (heads, deps, labels) = gerenate_remote(gold_tree)
            remote_head_list: [int] = []
            remote_dep_list: [int] = []
            remote_label_list: [LabelField] = []
            remote_nodes_span_list: [SpanField] = []
            for head_sublist, dep_sublist, label_sublist in zip(heads, deps, labels):
                for head, dep, label in zip(head_sublist, dep_sublist, label_sublist):
                    remote_head_list.append(head)
                    remote_dep_list.append(dep)
                    remote_label_list.append(LabelField(label, label_namespace="remote_labels"))

            for (start, end) in spans:
                # SUDA code use (i, i+1) to represent the span covering the token i,
                # while AllenNlp uses (i,i)
                end = end - 1
                remote_nodes_span_list.append(SpanField(start, end, sentence_field))

            if len(remote_head_list) == 0:  # In that case remote_dep_list and remote_label_list is also empty
                empty_array_field = ArrayField(np.zeros(1)).empty_field()
                empty_label_list_field = ListField([LabelField("dummy")]).empty_field()
                try:
                    empty_span_list_field = ListField([SpanField(0, 0, sentence_field)]).empty_field()
                except:
                    print(tokenized_text)
                fields["remote_heads"] = empty_array_field
                fields["remote_deps"] = empty_array_field
                fields["remote_labels"] = empty_label_list_field
                fields["remote_nodes_spans"] = empty_span_list_field
            else:
                fields["remote_heads"] = ArrayField(np.array(remote_head_list))
                fields["remote_deps"] = ArrayField(np.array(remote_dep_list))
                fields["remote_labels"] = ListField(remote_label_list)
                fields["remote_nodes_spans"] = ListField(remote_nodes_span_list)

        spans: List[Field] = []
        gold_labels = []
        for start, end in enumerate_spans(tokenized_text):
            spans.append(SpanField(start, end, sentence_field))
            # TODO: Use gold labels in the training instead of the oracle_label function. Right now they are needed
            #  for creating the vocabulary of labels
            if gold_tree is not None:
                # SUDA code use (i, i+1) to represent the span covering the token i,
                # while AllenNlp uses (i,i)
                gold_label = str(gold_primary_tree.oracle_label(start, end + 1))
                gold_labels.append(gold_label)

        span_list_field: ListField = ListField(spans)
        fields["spans"] = span_list_field

        if gold_tree is not None:
            fields["span_labels"] = SequenceLabelField(gold_labels, span_list_field)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # TODO: In the future, I would like to pass to the text_to_instance only a string.
        #  Training data comes pre-tokenized and the text may not. Need to think about it.
        #  Also Need to think how to support reading features from file.
        #  For both issues, maybe another reader is the solution?  Sound reasonable.
        file_list: List[Tuple[str, str]] = list(self._get_file_list(file_path))

        if self.shuffle:
            shuffle(file_list)

        for file_path, dataset_title in file_list:
            passage = file2passage(file_path)

            tokenized_text = [node.text for node in sorted(passage.layer("0").all, key=lambda x: x.position)]

            id = passage.ID
            assert id, "Attribute 'id' is required per passage when using this model"

            lang = passage.attrib.get("lang")
            assert lang, "Attribute 'lang' is required per passage when using this model"

            gold_tree = None
            if "1" in passage._layers:
                gold_tree = copy.deepcopy(passage)

            if self.filter_func and self.filter_func(tokenized_text):
                print(f'Filtering out passage {id}!')
                continue

            # TODO: Move it out of the class
            if len(tokenized_text) > 200:
                print(f'Filtering out passage {id} with length longer than 200 tokens!')
                continue

            # TODO: There is no need to give both the passage object and the tokenized_text, language and id.
            #  Also, if we support text-only input, you don't have these.
            yield self.text_to_instance(tokenized_text, passage, dataset_title, lang, id, gold_tree)

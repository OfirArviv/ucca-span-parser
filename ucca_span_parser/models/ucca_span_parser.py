from typing import Dict, List
import torch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, get_lengths_from_binary_sequence_mask
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric
from ucca.core import Passage

from ucca_span_parser.factories.allennlp_builtin import SpanExtractorFactory, Seq2VecEncoderFactory
from ucca_span_parser.htl_suda_ucca_parser.convert.convert import to_UCCA
from ucca_span_parser.htl_suda_ucca_parser.convert.trees import InternalParseNode
from ucca_span_parser.htl_suda_ucca_parser.module.span_parser import Span_Parser_Factory
from ucca_span_parser.htl_suda_ucca_parser.module.remote_parser import Remote_Parser_Factory
#from pytorch_memlab import profile



@Model.register('ucca-span-parser')
class UccaSpanParser(Model):
    def __init__(self,
                 token_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoderFactory,
                 span_decoder: Span_Parser_Factory,
                 span_extractor: SpanExtractorFactory,
                 remote_parser: Remote_Parser_Factory,
                 evaluator: Metric,
                 vocab: Vocabulary,
                 embedding_dropout: float = 0,
                 parse_remote: bool = True) -> None:
        super().__init__(vocab)
        self.token_embedder = token_embedder
        self.encoder = encoder(self.token_embedder.get_output_dim())
        self.span_extractor = span_extractor(self.encoder.get_output_dim())
        self.span_decoder = span_decoder(self.span_extractor.get_output_dim(), vocab)
        self.remote_parser = remote_parser(self.span_extractor.get_output_dim(), vocab)
        self.evaluator = evaluator
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.parse_remote = parse_remote

    #@profile
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                passage: List[Passage],
                dataset_label: List[str],
                spans: torch.Tensor,
                lang: torch.Tensor,
                id: List[str] = None,
                gold_ucca_tree: List[Passage] = None,
                gold_primary_tree: List[InternalParseNode] = None,
                span_labels: torch.Tensor = None,
                remote_heads: torch.Tensor = None,
                remote_deps: torch.Tensor = None,
                remote_labels: torch.Tensor = None,
                remote_nodes_spans: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.token_embedder(tokens, lang=lang)
        text_mask = get_text_field_mask(tokens)
        sentence_lengths = get_lengths_from_binary_sequence_mask(text_mask)
        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()
        embedded_text_input = self.embedding_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, text_mask)
        # span_representations.shape: torch.Size([1, 276, 20])
        span_representations = self.span_extractor(encoded_text, spans, text_mask, span_mask)

        predicted_tree = self.span_decoder.predict(span_representations, sentence_lengths)
        predict_passages = list(map(to_UCCA, passage, predicted_tree))
        if self.parse_remote:
            predict_passages = self.remote_parser.restore_remote(predict_passages, span_representations, sentence_lengths)
        output = {"prediction": predict_passages}

        if gold_ucca_tree is not None:
            tree_loss = self.span_decoder.get_loss(span_representations, sentence_lengths, gold_primary_tree)
            # Looking at the span start index is enough to know if
            # this is padding or not. Shape: (batch_size, num_spans)
            remote_loss = 0
            if self.parse_remote:
                remote_nodes_spans_mask = (remote_nodes_spans[:, :, 0] >= 0).squeeze(-1).long()
                remote_labels_mask = (remote_labels[:, :] >= 0).squeeze(-1).long()
                remote_loss = self.remote_parser.get_loss(span_representations, sentence_lengths, remote_nodes_spans,
                                                          remote_nodes_spans_mask, remote_heads, remote_deps, remote_labels,
                                                          remote_labels_mask)

            output["loss"] = tree_loss + remote_loss
            for i in range(len(predicted_tree)):
                self.evaluator(dataset_label[i], predict_passages[i], gold_ucca_tree[i])

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.evaluator.get_metric(reset)

from typing import Dict
import torch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn.util import get_text_field_mask, get_lengths_from_binary_sequence_mask
from allennlp.data import Vocabulary
from ucca.core import Passage
from htl_suda_ucca_parser import InternalParseNode, to_UCCA
from htl_suda_ucca_parser.module import Topdown_Span_Parser, Remote_Parser


@Model.register('ucca-span-parser')
class UccaSpanParser(Model):
    def __init__(self,
                 token_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 span_decoder: Topdown_Span_Parser,
                 span_extractor: SpanExtractor,
                 remote_parser: Remote_Parser,
                 evaluator: any,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.token_embedder = token_embedder
        self.encoder = encoder
        self.span_decoder = span_decoder
        self.span_extractor = span_extractor
        self.remote_parser = remote_parser
        self.evaluator = evaluator

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                passage: [Passage],
                dataset_label: [str],
                spans: torch.Tensor,
                lang: torch.Tensor,
                id: [str] = None,
                gold_ucca_tree: [Passage] = None,
                gold_primary_tree: [InternalParseNode] = None,
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

        encoded_text = self.encoder(embedded_text_input, text_mask)
        # span_representations.shape: torch.Size([1, 276, 20])
        span_representations = self.span_extractor(encoded_text, spans, text_mask, span_mask)

        predicted_tree = self.span_decoder.predict(span_representations, sentence_lengths)
        predict_passages = list(map(to_UCCA, passage, predicted_tree))
        predict_passages = self.remote_parser.restore_remote(predict_passages, span_representations, sentence_lengths)
        output = {"prediction": predict_passages}

        if gold_ucca_tree is not None:
            tree_loss = self.span_decoder.get_loss(span_representations, sentence_lengths, gold_primary_tree)
            # Looking at the span start index is enough to know if
            # this is padding or not. Shape: (batch_size, num_spans)
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

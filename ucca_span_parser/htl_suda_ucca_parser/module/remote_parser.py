"""
This code is taken from https://github.com/SUDA-LA/ucca-parser
"""
from abc import abstractmethod

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from torch import nn

from ..convert.trees import get_position
from .biaffine import Biaffine


class Remote_Parser(nn.Module):
    def __init__(self, vocab, span_dim, mlp_dim, dropout=0):
        super(Remote_Parser, self).__init__()
        self.vocab: Vocabulary = vocab
        num_layers = 1
        self.label_head_mlp = FeedForward(span_dim, num_layers, mlp_dim, torch.relu,
                                          dropout)
        self.label_dep_mlp = FeedForward(span_dim, num_layers, mlp_dim, torch.relu,
                                         dropout)

        self.label_biaffine = Biaffine(
            mlp_dim, vocab.get_vocab_size("remote_labels"), bias_dep=True, bias_head=True
        )

    def forward(self, span_vectors):
        label_head_mlp_out = self.label_head_mlp(span_vectors)
        label_dep_mlp_out = self.label_dep_mlp(span_vectors)

        label_scores = self.label_biaffine(label_head_mlp_out, label_dep_mlp_out)
        return label_scores

    def score(self, span_vectors, sen_len, all_span):
        span_vectors = [span_vectors[get_position(sen_len, i, j)] for i, j in all_span]
        span_vectors = torch.stack(span_vectors)
        label_scores = self.forward(span_vectors.unsqueeze(0))
        return label_scores.squeeze(0).permute(1, 2, 0)

    def get_loss(self, spans, sen_lens, remote_nodes_spans, remote_nodes_spans_mask, remote_heads, remote_deps,
                 remote_labels, remote_labels_mask):
        loss_func = torch.nn.CrossEntropyLoss()
        batch_loss = []
        remote_label_length = get_lengths_from_binary_sequence_mask(remote_labels_mask)
        remote_nodes_spans_length = get_lengths_from_binary_sequence_mask(remote_nodes_spans_mask)
        if len(remote_label_length.shape) == 0:
            batch_loss.append(torch.zeros(1))
        else:
            for i, length in enumerate(sen_lens):
                if remote_label_length[i] == 0:
                    continue
                span_num = (1 + length) * length // 2
                label_scores = self.score(spans[i][:span_num], length,
                                          remote_nodes_spans[i][:remote_nodes_spans_length[i]])
                batch_loss.append(
                    loss_func(
                        label_scores[remote_heads[i][:remote_label_length[i]].long(),
                                     remote_deps[i][:remote_label_length[i]].long()],
                        remote_labels[i][:remote_label_length[i]].to(spans[i].device),
                    )
                )
        # TODO: Changing mean() to sum() because that how its done in the original code
        return torch.stack(batch_loss, 0).sum()

    def predict(self, span, sen_len, all_nodes, remote_head):
        label_scores = self.score(span, sen_len, all_nodes)
        try:
            labels = label_scores[remote_head].argmax(dim=-1)
        except:
            print(label_scores)
        return labels

    def restore_remote(self, passages, spans, sen_lens):
        def get_span_index(node):
            terminals = node.get_terminals()
            return (terminals[0].position - 1, terminals[-1].position)

        for passage, span, length in zip(passages, spans, sen_lens):
            heads = []
            nodes = passage.layer("1").all
            ndict = {node: i for i, node in enumerate(nodes)}
            span_index = [get_span_index(i) for i in nodes]
            for node in nodes:
                for edge in node._incoming:
                    if "-remote" in edge.tag:
                        heads.append(node)
                        if hasattr(edge, "categories"):
                            edge.categories[0]._tag = edge.categories[0]._tag.strip(
                                "-remote"
                            )
                        else:
                            edge._tag = edge._tag.strip("-remote")
            heads = [ndict[node] for node in heads]

            if len(heads) == 0:
                continue
            else:
                label_scores = self.predict(span, length, span_index, heads)

            for head, label_score in zip(heads, label_scores):
                for i, score in enumerate(label_score):
                    label = self.vocab.get_token_from_index(score.item(), "remote_labels")
                    if label is not "<NULL>" and not nodes[i]._tag == "PNCT":
                        passage.layer("1").add_remote(nodes[i], label, nodes[head])
        return passages


class Remote_Parser_Factory(Registrable):
    @abstractmethod
    def __call__(self, span_dim: int, vocab: Vocabulary) -> Remote_Parser:
        raise NotImplemented


@Remote_Parser_Factory.register("basic")
class Basic_Remote_Parser_Factory(Remote_Parser_Factory):
    def __init__(self, mlp_dim: int, dropout=0):
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def __call__(self, span_dim: int, vocab: Vocabulary) -> Remote_Parser:
        return Remote_Parser(vocab, span_dim, self.mlp_dim, self.dropout)


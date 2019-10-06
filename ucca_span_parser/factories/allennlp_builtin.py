from abc import abstractmethod
from allennlp.common import Registrable
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.span_extractors import BidirectionalEndpointSpanExtractor
from torch import nn


class SpanExtractorFactory(Registrable):
    @abstractmethod
    def __call__(self, input_dim):
        raise NotImplemented


@SpanExtractorFactory.register("bidirectional_endpoint")
class BidirectionalEndpointSpanExtractorFactory(SpanExtractorFactory):
    def __call__(self, input_dim):
        return BidirectionalEndpointSpanExtractor(input_dim=input_dim)


class Seq2VecEncoderFactory(Registrable):
    @abstractmethod
    def __call__(self, input_size: int):
        raise NotImplemented


@Seq2VecEncoderFactory.register("lstm")
class LstmEncoderFactory(Seq2VecEncoderFactory):
    @abstractmethod
    def __init__(self, hidden_size: int, num_layers: int = 1, bias: bool = True, dropout: float = 0,
                 bidirectional: bool = False):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

    def __call__(self, input_size: int):
        return PytorchSeq2SeqWrapper(nn.LSTM(input_size, self.hidden_size, self.num_layers, self.bias, batch_first=True,
                                             dropout=self.dropout, bidirectional=self.bidirectional))

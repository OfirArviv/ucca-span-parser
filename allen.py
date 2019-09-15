import itertools
import logging
import os
import torch
from allennlp.common.util import prepare_global_logging
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules import Embedding, Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.span_extractors import SpanExtractor, BidirectionalEndpointSpanExtractor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.training import Trainer
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PosTagIndexer, PretrainedBertIndexer, DepLabelIndexer, \
    NerTagIndexer
from torch import nn, optim


from htl_suda_ucca_parser.module import Topdown_Span_Parser, Remote_Parser
from models import UccaSpanParser
from metrics import UccaScores
from tokenizers import SpacyMultilingualWhitespaceWordSplitter
from dataset_readers import UccaSpanParserDatasetReader

from token_indexers import LanguageIndexer


"""
    elmo_indexer = ELMoTokenCharactersIndexer()

    aligning_files = {
        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_best_mapping.pth",
        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_best_mapping.pth",
        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_best_mapping.pth"
    }
    options_files = {
        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json"
    }
    weight_files = {
        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_weights.hdf5",
        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_weights.hdf5",
        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_weights.hdf5"
    }
    scalar_mix_parameters = [-9e10, 1, -9e10]
    elmo_embedder = ElmoTokenEmbedderMultiLang(options_files=options_files,
                                               weight_files=weight_files,
                                               dropout=0.3,
                                               requires_grad=False,
                                               scalar_mix_parameters=scalar_mix_parameters,
                                               aligning_files=aligning_files)
                                               
     word_embedder = BasicTextFieldEmbedder({"elmo": elmo_embedder,
                                            "deps": linguistic_features_embedding,
                                            "ner": linguistic_features_embedding,
                                            "pos": linguistic_features_embedding,
                                            "lang": linguistic_features_embedding,
                                            })
    """

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)

    #prepare_global_logging(serialization_dir, file_friendly_logging)

    bert_mode = "bert-base-multilingual-cased"

    linguistic_features_embedding_dim = 50
    encoder_output_dim = 200
    remote_parser_mlp_dim = 100

    batch_size = 10

    train_dataset_folder = "C:/Users/t-ofarvi/PycharmProjects/UCCA_Dataset_29-06-09/tryout"  # "C:/Users/t-ofarvi/Desktop/train_allen"
    validation_dataset_folder = "C:/Users/t-ofarvi/Desktop/dev_allen"

    model_dir = "C:/Users/t-ofarvi/PycharmProjects/UCCA_Dataset_29-06-09/tryout-model"
    vocab_dir = f'{model_dir}/vocabulary'

    word_tokenizer = SpacyMultilingualWhitespaceWordSplitter()
    # NOTE: The word tokenizer is a SpaCy tokenizer, which is a little different from the BERT tokenizer.
    # This was done for convince.
    bert_indexer = PretrainedBertIndexer(
        pretrained_model=bert_mode,
        do_lowercase="uncased" in bert_mode,
        truncate_long_sequences=False
    )

    word_indexer = {"bert": bert_indexer,
                    "deps": DepLabelIndexer(namespace="deps_tags"),
                    "ner": NerTagIndexer(),
                    "pos": PosTagIndexer(),
                    "lang": LanguageIndexer()}

    train_ds, validation_ds = (UccaSpanParserDatasetReader(word_tokenizer, word_indexer).read(folder) for
                               folder in [train_dataset_folder, validation_dataset_folder])

    if os.path.exists(vocab_dir):
        vocab = Vocabulary.from_files(vocab_dir)
    else:
        vocab = Vocabulary.from_instances(itertools.chain(train_ds, validation_ds))
        vocab.save_to_files(vocab_dir)

    vocab_namespaces = vocab._index_to_token.keys()
    max_vocab_size = max([vocab.get_vocab_size(namespace) for namespace in vocab_namespaces])

    iterator = BucketIterator(batch_size=batch_size,
                              # This is for testing. To see how big of batch size the GPU can handle.
                              biggest_batch_first=True,
                              sorting_keys=[("tokens", "num_tokens")],
                              )
    iterator.index_with(vocab)

    linguistic_features_embedding = Embedding(num_embeddings=max_vocab_size + 2,
                                              embedding_dim=linguistic_features_embedding_dim,
                                              # padding_index=0 I do not understand what is does
                                              )
    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-multilingual-cased",
        top_layer_only=False,
        requires_grad=False,
    )

    word_embedder = BasicTextFieldEmbedder({"bert": bert_embedder,
                                            "deps": linguistic_features_embedding,
                                            "ner": linguistic_features_embedding,
                                            "pos": linguistic_features_embedding,
                                            "lang": linguistic_features_embedding,
                                            },
                                           {"bert": {"input_ids": "bert",
                                                     "offsets": "bert-offsets"},
                                            "deps": {"inputs": "deps"},
                                            "ner": {"inputs": "ner"},
                                            "pos": {"inputs": "pos"},
                                            "lang": {"inputs": "lang"}},
                                           allow_unmatched_keys=True)

    encoder: Seq2VecEncoder = PytorchSeq2SeqWrapper(nn.LSTM(word_embedder.get_output_dim(), encoder_output_dim, 2,
                                                            dropout=0.4, bidirectional=True, batch_first=True))

    # span_extractor: SpanExtractor = SelfAttentiveSpanExtractor(input_dim=encoder.get_output_dim())
    span_extractor: SpanExtractor = BidirectionalEndpointSpanExtractor(input_dim=encoder.get_output_dim())

    span_decoder = Topdown_Span_Parser(vocab, span_extractor.get_output_dim())

    remote_parser = Remote_Parser(vocab, span_extractor.get_output_dim(), remote_parser_mlp_dim)

    model = UccaSpanParser(
        word_embedder,
        encoder,
        span_decoder,
        span_extractor,
        remote_parser,
        UccaScores(),
        vocab
    )

    if torch.cuda.is_available():
        cuda_device = list(range(torch.cuda.device_count()))
        model = model.cuda(cuda_device[0])
    else:
        cuda_device = -1

    trainer = Trainer(
        model=model,
        optimizer=optim.Adam(model.parameters()),
        iterator=iterator,
        train_dataset=train_ds,
        validation_dataset=validation_ds,
        validation_metric="+labeled_average_F1",
        patience=10,
        cuda_device=cuda_device,
        num_epochs=50,
        serialization_dir=model_dir,
        grad_norm=5.0
    )

    metrics = trainer.train()

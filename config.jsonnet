local bert_model = "bert-base-multilingual-cased";
local bert_do_lowercase = False;

local linguistic_features_embedding_dim = 50;
local encoder_output_dim = 200;
local remote_parser_mlp_dim = 100;
local batch_size = 50;
local num_epochs = 1000;
local patience = 10;



local word_embedding_dim = 5;
local char_embedding_dim = 3;
local embedding_dim = word_embedding_dim + char_embedding_dim;
local hidden_dim = 6;
local num_epochs = 1000;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'C:/Users/t-ofarvi/PycharmProjects/UCCA_Dataset_29-06-09/tryout',
    "validation_data_path": 'C:/Users/t-ofarvi/PycharmProjects/UCCA_Dataset_29-06-09/tryout-validation',
    "dataset_reader": {
        "type": "ucca-span",
        "word_tokenizer": "spacy-whitespace-multilingual",
        "token_indexers": {
            "bert": { "type": "bert-pretrained",
                       "pretrained_model": bert_model,
                       "do_lowercase": bert_do_lowercase,
                       "truncate_long_sequences": False },
             "deps": {"type": "dependency_label",
                      "namespace": "deps_tags"},
             "ner": { "type": "ner_tag" },
             "pos": {"type": "pos_tag" },
             "lang": {"type": "language" }
        }
    },
    "model": {
        "type": "ucca-span-parser",
        "token_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "top_layer_only": False,
                    "requires_grad": False
                },
                "deps": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim
                },
                "ner": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim
                },
                "pos": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim
                },
                "lang": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim
                },
            "embedder_to_indexer_map": {
                "bert": {
                    "input_ids": "bert",
                    "offsets": "bert-offsets"
                    },
                "deps": {"inputs": "deps"},
                "ner": {"inputs": "ner"},
                "pos": {"inputs": "pos"},
                "lang": {"inputs": "lang"}
                },
            "allow_unmatched_keys": True
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim,
            "dropout": 0.4,
            "bidirectional": True,
            "batch_first": True
        },
        "span_decoder":{
            "type": "topdown",


        },
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["tokens", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}
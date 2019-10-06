local bert_model = "bert-base-multilingual-cased";
local bert_do_lowercase = false;

local linguistic_features_embedding_dim = 50;
local linguistic_features_num_embeddings = 1000;
local encoder_output_dim = 200;
local remote_parser_mlp_dim = 100;

local batch_size = 10;
local num_epochs = 30;
local patience = 10;
local grad_norm = 5.0;
local learning_rate = 0.1;
local cuda_device = 0;

{
    "train_data_path": "/cs/labs/oabend/ofir.arviv/train_dev-data-17.9_lang/train_allen",
    "validation_data_path": "/cs/labs/oabend/ofir.arviv/train_dev-data-17.9_lang/dev_allen",
    "test_data_path": "/cs/labs/oabend/ofir.arviv/sharetask_test/test_allen",
    "evaluate_on_test": true,
    "dataset_reader": {
        "type": "ucca-span",
        "word_tokenizer": "spacy-whitespace-multilingual",
        "token_indexers": {
            "bert": { "type": "bert-pretrained",
                       "pretrained_model": bert_model,
                       "do_lowercase": bert_do_lowercase,
                       "truncate_long_sequences": false },
             "deps": {"type": "dependency_label",
                      "namespace": "deps_tags"},
             "ner": { "type": "ner_tag" },
             "pos": {"type": "pos_tag" },
             "lang": {"type": "language" }
        },
        "shuffle": true,
        "lazy": true
    },
    "model": {
        "type": "ucca-span-parser",
        "token_embedder": {
            "type": "basic",
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "top_layer_only": false,
                    "requires_grad": false
                },
                "deps": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim,
                    "num_embeddings": linguistic_features_num_embeddings,
                },
                "ner": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim,
                    "num_embeddings": linguistic_features_num_embeddings,
                },
                "pos": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim,
                    "num_embeddings": linguistic_features_num_embeddings,
                },
                "lang": {
                    "type": "embedding",
                    "embedding_dim": linguistic_features_embedding_dim,
                    "num_embeddings": linguistic_features_num_embeddings,
                },
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
            "allow_unmatched_keys": true
        },
        "encoder": {
            "type": "lstm",
            "hidden_size": encoder_output_dim,
            "num_layers": 2,
            "dropout": 0.4,
            "bidirectional": true,
        },
        "span_extractor": "bidirectional_endpoint",
        "span_decoder": "topdown",
        "remote_parser":{
            "type": "basic",
            "mlp_dim": remote_parser_mlp_dim
        },
        "evaluator":{
            "type": "ucca-scores"
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["tokens", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": "adam",
        "patience": patience,
        "grad_norm": grad_norm,
        "cuda_device": cuda_device,
        "validation_metric": "+labeled_average_F1"
    }
}
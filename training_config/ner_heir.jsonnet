{
    "dataset_reader": {
        "type": "conll_03_reader",
        "token_indexers": {
          "tokens": {
            "type": "single_id",
            "namespace": "tokens",
            "lowercase_tokens": true
          },
          "characters":{
              "type": "characters",
              "namespace": "character_tokens",
              "min_padding_length": 3
          }
        }
      },
    "train_data_path": "./data/train.txt",
    "validation_data_path": "./data/validation.txt",
    "model":{
        "type": "heir_lstm",
        "use_crf": true,
            "word_embedder":{
                "token_embedders":{
                "tokens":{
                    "type": "embedding",
                    "pretrained_file": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
                    "embedding_dim": 50,
                    "trainable": false
                }
                }
            },
            "character_embedder":{
                "token_embedders":{
                "characters":{
                    "type": 'embedding',
                    "embedding_dim": 16,
                    "trainable": true
                }
                }
            },
            "character_encoder":{
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 128,
                "ngram_filter_sizes": [3],
                "conv_layer_activation": "relu"
            },
        "encoder":{
            "type": "lstm",
            "input_size": 50 + 128,
            "hidden_size": 200,
            "bidirectional": true,
            "dropout": 0.1,
            "num_layers": 2
        }
    },
    "data_loader": {
        // See http://docs.allennlp.org/master/api/data/dataloader/ for more info on acceptable
        // parameters here.
        "batch_size": 64,
        "shuffle": true
    },
    "trainer":{
        "num_epochs": 10,
        "patience": 3,
        "grad_clipping": 5.0,
        "validation_metric": "-loss",
        "optimizer":{
            "type":"adam",
            "lr": 0.003
        }
    }
}
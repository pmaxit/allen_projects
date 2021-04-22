{
    "dataset_reader": {
        "type": "conll_03_reader",
        "token_indexers": {
          "tokens": {
            "type": "single_id",
            "namespace": "tokens",
            "lowercase_tokens": true
          }
        }
      },
    "train_data_path": "./data/train.txt",
    "validation_data_path": "./data/validation.txt",
    "model":{
        "type": "ner_lstm",
        "embedder":{
            "token_embedders":{
                "tokens":{
                    "type": "embedding",
                    "pretrained_file": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
                    "embedding_dim": 50,
                    "trainable": false

                }
            }
        },
        "encoder":{
            "type": "lstm",
            "input_size": 50,
            "hidden_size": 25,
            "bidirectional": true
        }
    },
    "data_loader": {
        // See http://docs.allennlp.org/master/api/data/dataloader/ for more info on acceptable
        // parameters here.
        "batch_size": 8,
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
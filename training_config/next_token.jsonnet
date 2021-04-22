{
    "dataset_reader":{
        "type": "name_dataset"
    },

    "train_data_path": "./data/first_names.all.txt",
    "model":{
        "type": "next_token_model",
        "text_field_embedder": {
            "token_embedders":{
                "characters":{
                    "type": "embedding",
                    "embedding_dim": 50,
                    "vocab_namespace": "character_vocab"
                }
            },
        },
        "contextualizer":{
            "type": "lstm",
            "num_layers":2,
            "input_size": 50,
            "hidden_size": 50,
            "dropout": 0.5
        }
    },
    "data_loader":{
        "batch_size":32,
        "shuffle":true
    },
    "trainer":{
        "num_epochs": 10,
        "patience": 3,
        "grad_clipping": 5.0,
        "optimizer":{
            "type":"adamw",
            "lr": 0.5
        },
        "callbacks": [{
            "type": "model_logger",
            "should_log_inputs": true
        }]
    }
}
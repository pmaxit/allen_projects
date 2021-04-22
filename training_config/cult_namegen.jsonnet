{
    "dataset_reader":{
        "type": "name_cultgen"
    },

    "train_data_path": "./data/names",
    "model":{
        "type": "next_token_model2",
        "text_field_embedder": {
            "token_embedders":{
                "characters":{
                    "type": "embedding",
                    "embedding_dim": 110,
                    "vocab_namespace": "tokens"
                }
            },
        },
        "label_embedder":{
            "type": "embedding",
            "embedding_dim": 18,
            "vocab_namespace": "labels"
        },
        "contextualizer":{
            "type": "lstm",
            "num_layers":2,
            "input_size": 50,
            "hidden_size": 50,
            "dropout": 0.5
        },
        "hidden_size": 512,
        "dropout": 0.5,
        "num_layers": 2
    },
    "data_loader":{
        "batch_size":16,
        "shuffle":true
    },
    "trainer":{
        "num_epochs": 50,
        "patience": 3,
        "grad_clipping": 0.25,
        "optimizer":{
            "type":"adamw",
            "lr": 0.1
        },

        "callbacks":[
            {
                "type": "tensorboard",
                "serialization_dir": "tb_logs"
            }
        ]
    }
}

import json

class Config:

    DEFAULT_CONFIG = {
        "experiment_name": "demo",
        "learning_rate": 0.001,
        "protein_model_name": "facebook/esm2_t36_3B_UR50D",
        "num_target_encoders_tuned": 2,
        "featurized_protein": False,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "dropout": 0.1,
        "lora_r": 4,
        "dataset_csv_path": "data/demo_dataset.csv",
        "protein_sequence_column": "target_sequence",
        "drug_smiles_column": "smiles",
        "target_columns": ["IC50"],
        "train_ratio":0.8,
        "val_ratio":0.1,
        "batch_size": 32,
        "num_workers": 4,
        "num_epochs": 10,
        "num_gpus": 1,
        "num_classes": 1,
        "seed": 42,
        "task_type": "classification",
        "evidential": False
        }

    def __init__(self, json_file):
        self.__dict__ = self.DEFAULT_CONFIG

        json_loaded = json.load(open(json_file))
        for key, value in json_loaded.items():
            self.__dict__[key] = value

    # value = config['param1'] とかで取得できるようにする
    def __getitem__(self, key):
        return self.__dict__[key]

    # config.param1 とかで取得できるようにする
    def __getattr__(self, key):
        return self.__dict__[key]

    # configの状態を確認できるようにする
    def __str__(self):
        output = "========config========\n"
        for key, value in self.__dict__.items():
            output += f"{key}: {value}\n"
        output += "======================"
        return output
    
    def to_dict(self):
        return self.__dict__

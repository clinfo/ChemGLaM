import json


class Config:

    DEFAULT_CONFIG = {"learning_rate": 0.001, 
                    "protein_model_name": "facebook/esm1_t34_670M_UR50S", 
                    "num_target_encoders_tuned": 2, 
                    "lora_alpha": 32, 
                    "lora_dropout": 0.1, 
                    "r": 4
                    }

    def __init__(self, json_file):
        self.__dict__ = self.DEFAULT_CONFIG

        json_loaded = json.load(open(json_file))
        for key, value in json_loaded.items():
            self.__dict__[key] = value

    #value = config['param1'] とかで取得できるようにする
    def __getitem__(self, key):
        return self.__dict__[key]
    
    #config.param1 とかで取得できるようにする
    def __getattr__(self, key):
        return self.__dict__[key]
    
    #configの状態を確認できるようにする
    def __str__(self):
        output = "========config========\n"
        for key, value in self.__dict__.items():
            output += f"{key}: {value}\n"
        output += "======================"
        return output


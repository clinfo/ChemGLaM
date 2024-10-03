
import lightning as L
import torch
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from transformers import AutoModelForSequenceClassification
from src.utils.config import Config
from src.model.layers import MLPNet


class ChemGLaM(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.save_hyperparameters(config)

        self.protein_model_name = config.protein_model_name

        self.drug_encoder = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.target_encoder = AutoModel.from_pretrained(self.protein_model_name)

        self.mlp_net = MLPNet()

        num_target_encoder_layers = int(self.protein_model_name.split("_")[1].split("t")[1])  # facebook/esm2_t30_150M_UR50D => 30
        self.num_target_encoders_tuned = 2  # TODO: configで指定

        lora_target_modules = [
            [f"esm.encoder.layer.{i}.attention.self.query" for i in range(num_target_encoder_layers - self.num_target_encoders_tuned, num_target_encoder_layers)],
            [f"esm.encoder.layer.{i}.attention.self.key" for i in range(num_target_encoder_layers - self.num_target_encoders_tuned, num_target_encoder_layers)],
            [f"esm.encoder.layer.{i}.attention.self.value" for i in range(num_target_encoder_layers - self.num_target_encoders_tuned, num_target_encoder_layers)],
        ]

        self.lora_target_modules = [item for sublist in lora_target_modules for item in sublist]

        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=self.lora_target_modules,
            inference_mode=False,
            r=4,  # TODO: configで指定
            lora_alpha=32  # TODO: configで指定
            lora_dropout=0.1  # TODO: configで指定
        )

        self.target_encoder = self.set_lora_config(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        self.learning_rate = config.learning_rate

    def forward(self, drug, target):

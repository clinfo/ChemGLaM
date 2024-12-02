
import lightning as L
import torch
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from src.utils.config import Config
from src.model.cross_attention import CrossAttention
from src.model.layers import MLPNet

class ChemGLaM(L.LightningModule):

    protein_dim_table = {
        "facebook/esm2_t48_15B_UR50D": 5120,
        "facebook/esm2_t36_3B_UR50D": 2560,
        "facebook/esm2_t33_650M_UR50D": 1280,
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t12_35M_UR50D": 480,
        "facebook/esm2_t6_8M_UR50D": 320,
    }

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.save_hyperparameters(config.to_dict())

        self.protein_model_name = config.protein_model_name

        self.drug_encoder = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).train() # MoLFormerはfloat型でないとエラーが出る
        
        if config.featurization_type == "embedding":
            self.target_encoder = None
        else:       
            self.target_encoder = AutoModel.from_pretrained(
                self.protein_model_name).half()

        self.cross_attention = CrossAttention(drug_dim=768,
                                              target_dim=self.protein_dim_table[self.protein_model_name],
                                              heads=8,
                                              dim_head=96,
                                              skip_connection=True)

        self.mlp_net = MLPNet(emb_dim=768+self.protein_dim_table[self.protein_model_name],
                              num_classes=config.num_classes, 
                              dropout=config.dropout) 

        num_target_encoder_layers = int(self.protein_model_name.split(
            "_")[1].split("t")[1])  # facebook/esm2_t30_150M_UR50D => 30
        
        self.num_target_encoders_tuned = config.num_target_encoders_tuned

        assert (config.featurization_type != "embedding") or (self.num_target_encoders_tuned < 1), \
            'LoRA can not be used when featurization_type is "embedding"'
            
        if config.num_target_encoders_tuned >= 1:
            lora_target_modules = [
                [f"encoder.layer.{i}.attention.self.query" for i in range(
                    num_target_encoder_layers - self.num_target_encoders_tuned, num_target_encoder_layers)],
                [f"encoder.layer.{i}.attention.self.key" for i in range(
                    num_target_encoder_layers - self.num_target_encoders_tuned, num_target_encoder_layers)],
                [f"encoder.layer.{i}.attention.self.value" for i in range(
                    num_target_encoder_layers - self.num_target_encoders_tuned, num_target_encoder_layers)],
            ]
            
            self.lora_target_modules = [
                item for sublist in lora_target_modules for item in sublist]

            self.lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=self.lora_target_modules,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            )
            
            self.target_encoder = self.set_lora_config(
                self.target_encoder, self.lora_config)
            self.target_encoder.print_trainable_parameters()
        else:
            if self.target_encoder is not None:  
                self.target_encoder.eval()
                for param in self.target_encoder.parameters():
                    param.requires_grad = False
                
        if config.task_type == "classification" and not config.evidential:
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif config.task_type == "classification" and config.evidential:
            self.loss = EvidentialLoss()
        elif config.task_type == "regression":
            self.loss = torch.nn.MSELoss()
    
        self.learning_rate = config.learning_rate
    
    def set_lora_config(self, model, lora_config: LoraConfig):
        for name, module in model.named_modules():
            if name not in lora_config.target_modules:
                module.requires_grad = False
            if "classifier" in name:
                module.requires_grad = True
        lora_model = get_peft_model(model, lora_config)
        return lora_model

    def forward(self, batch):
        drug_ids = batch["drug_ids"]
        drug_mask = batch["drug_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]
        
        drug_output = self.drug_encoder(
            input_ids=drug_ids, attention_mask=drug_mask).last_hidden_state
        
        if self.target_encoder is not None:
            target_output = self.target_encoder(
                input_ids=target_ids, attention_mask=target_mask).last_hidden_state
        else:
            target_output = batch["target_embedding"]
        
        interaction_output, weight = self.cross_attention(
            drug_output, target_output, drug_mask, target_mask)
        
        input_mask_expanded = drug_mask.unsqueeze(-1).expand(interaction_output.size()).float()
        sum_embeddings = torch.sum(interaction_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        interaction_output = sum_embeddings / sum_mask
 
        target_output = target_output * target_mask.unsqueeze(-1)
        target_output = target_output.sum(dim=1) / target_mask.sum(dim=1).unsqueeze(-1)
        
        interaction_output = torch.cat([interaction_output, target_output], dim=1)
        
        output = self.mlp_net(interaction_output)

        return output, weight

    
    def training_step(self, batch, batch_idx):
        output, _ = self(batch)
        measures = batch["measures"]
        loss = self.loss(output, measures)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output, _ = self(batch)
        measures = batch["measures"]
        loss = self.loss(output, measures)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

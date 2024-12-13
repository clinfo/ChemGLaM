import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    """
    drug: (batch_size, drug_len, drug_dim)
    target: (batch_size, target_len, target_dim)
    calculate attention score between drug and target
    """

    def __init__(self, drug_dim=768, target_dim=2560, heads=8, dim_head=96, skip_connection=True):
        super().__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(drug_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(target_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(target_dim, heads * dim_head, bias=False)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=heads*dim_head, 
                                                    num_heads=heads,
                                                    bias=False, 
                                                    batch_first=True)
        self.to_out = nn.Linear(heads * dim_head, drug_dim)
        self.skip_connection = skip_connection
        self.layer_norm = nn.LayerNorm(drug_dim)

    def forward(self, drug, target, drug_mask, pro_mask):
        b, n, _, h = *drug.shape, self.heads
        
        q = self.to_q(drug).view(b, n, -1)
        target_len = target.shape[1]
        k = self.to_k(target).view(b, target_len, -1)
        v = self.to_v(target).view(b, target_len, -1)
        
        key_padding_mask = ~pro_mask.bool()        
        attn_output, attn_weights = self.multihead_attn(q, k, v, key_padding_mask=key_padding_mask)

        out = self.to_out(attn_output)
        if self.skip_connection:
            out = self.layer_norm(out + drug)
            
        return out, attn_weights
import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    """
    drug: (batch_size, drug_len, drug_dim)
    target: (batch_size, target_len, target_dim)
    calculate attention score between drug and target
    """

    def __init__(self, drug_dim=768, target_dim=2560, heads=8, dim_head=96):
        super().__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.heads = heads
        
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(drug_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(target_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(target_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, drug_dim)
        self.layer_norm = nn.LayerNorm(drug_dim)

    def forward(self, drug, target, drug_mask, pro_mask):
        b, n, _, h = *drug.shape, self.heads
        
        # Project drug into query space
        q = self.to_q(drug).view(b, n, self.heads, -1).transpose(1, 2)
        
        # Project target into key and value space
        target_len = target.shape[1]
        k = self.to_k(target).view(b, target_len, self.heads, -1).transpose(1, 2)
        v = self.to_v(target).view(b, target_len, self.heads, -1).transpose(1, 2)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        drug_mask = drug_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.heads, n, target_len)
        masked_dots = dots.masked_fill(drug_mask == 0, -1e6)
        pro_mask = pro_mask.unsqueeze(1).unsqueeze(-2).expand(-1, self.heads, n, target_len)
        masked_dots = masked_dots.masked_fill(pro_mask == 0, -1e6)
        
        # Apply softmax to compute attention weights
        attn = F.softmax(masked_dots, dim=-1)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)

        out = self.to_out(out)
        
        assert drug.shape == out.shape, "Shape mismatch between drug and out"
        out = self.layer_norm(out + drug)
            
        return out, attn
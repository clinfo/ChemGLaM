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
        self.to_out = nn.Linear(heads * dim_head, drug_dim)
        self.skip_connection = skip_connection
        self.layer_norm = nn.LayerNorm(drug_dim)

    def forward(self, drug, target, drug_mask, pro_mask):
        b, n, _, h = *drug.shape, self.heads
        q = self.to_q(drug).view(b, n, h, -1).transpose(1, 2)

        target = target.float()
        target_len = target.shape[1]
        try:
            k = self.to_k(target).view(b, target_len, h, -1).transpose(1, 2)
            v = self.to_v(target).view(b, target_len, h, -1).transpose(1, 2)
        except:
            print('target:', target.shape)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = F.softmax(dots, dim=-1)

        mask = drug_mask.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, h, n, target_len)
        attn = attn.masked_fill(mask == 0, 0)

        mask = pro_mask.unsqueeze(1)
        mask = mask.unsqueeze(-2)
        mask = mask.expand(-1, h, n, target_len)
        attn = attn.masked_fill(mask == 0, 0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)

        out = self.to_out(out)
        if self.skip_connection:
            out = self.layer_norm(out + drug)

        return out, attn

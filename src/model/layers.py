from torch import nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    """_summary_

    Args:
    - inter_embed_dim: int, the dimension of the intermediate embeddings   
    - num_classes: int, the number of classes to predict
    - dropout: float, the dropout rate

    Returns:
    - z: tensor, the output of the network
    """

    def __init__(self, inter_embed_dim, num_classes, dropout=0.2):

        super().__init__()
        self.desc_skip_connection = True
        print('dropout is {}'.format(dropout))

        self.fc1 = nn.Linear(inter_embed_dim, inter_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(inter_embed_dim, inter_embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(inter_embed_dim, num_classes)

    def forward(self, inter_emb):
        x_out = self.fc1(inter_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        x_out = x_out + inter_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        z = self.final(z + x_out)

        return z


class LMLayer(nn.Module):
    """_summary_

    Args:
    - n_embd: int, the dimension of the embeddings
    - n_vocab: int, the number of classes to predict

    Returns:
    - tensor: tensor, the output of the network
    """

    def __init__(self, n_embd, n_vocab):
        super().__init__()
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor

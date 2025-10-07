from torch import nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    """_summary_

    Args:
    - emb_dim: int, the dimension of the intermediate embeddings
    - num_classes: int, the number of classes to predict
    - dropout: float, the dropout rate
    - num_layers: int, the number of MLP layers (default: 2)

    Returns:
    - z: tensor, the output of the network
    """

    def __init__(self, emb_dim, num_classes, dropout=0.2, num_layers=2):

        super().__init__()
        self.desc_skip_connection = True
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        print('dropout is {}'.format(dropout))
        print('num_layers is {}'.format(num_layers))

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(emb_dim, emb_dim))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.GELU())

        self.final = nn.Linear(emb_dim, num_classes)

    def forward(self, inter_emb):
        x_out = inter_emb
        x_residual = inter_emb

        for i in range(self.num_layers):
            # Linear layer
            x_out = self.layers[i * 3](x_out)
            # Dropout
            x_out = self.layers[i * 3 + 1](x_out)
            # GELU
            x_out = self.layers[i * 3 + 2](x_out)
            # Residual connection
            x_out = x_out + x_residual
            x_residual = x_out

        z = self.final(x_out)
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

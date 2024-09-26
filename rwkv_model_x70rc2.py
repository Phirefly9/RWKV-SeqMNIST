import torch
from core_rwkv_x070rc2 import RWKV
import types

class RwkvModel(torch.nn.Module):

    def __init__(self, input_scan_dim, output_dim):
        super(RwkvModel, self).__init__()
        self.input_scan_dim = input_scan_dim

        tmp = types.SimpleNamespace()
        tmp.n_layer = 2
        tmp.n_embd = 64
        tmp.head_size_a = 64 # don't change
        tmp.head_size_divisor = 8 # don't change
        self.gpt_config = tmp
        num_fc_layers = 2
        num_fc_dims = 64

        self.encoder = torch.nn.Linear(input_scan_dim, tmp.n_embd)
        # self.encoder2 = torch.nn.Linear(tmp.n_embd, tmp.n_embd)
        self.rwkv = RWKV(self.gpt_config)
        
        self.readout = torch.nn.Linear(tmp.n_embd, output_dim)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        batch_size = x.size(0)
        state = torch.zeros(batch_size, self.gpt_config.n_layer * (2+self.gpt_config.head_size_a), self.gpt_config.n_embd).to(x.device)
        x = x.squeeze(1)

        
        x = self.encoder(x)
        x = torch.relu(x)
        x = torch.square(x)
        # x = self.encoder2(x)
        # x = torch.relu(x)
        # x = torch.square(x)


        x, state = self.rwkv(x, state)
        return self.readout(x[:, -1, :])

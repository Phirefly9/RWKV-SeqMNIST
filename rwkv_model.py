import torch
from core_rwkv import RWKV
import types

class RwkvModel(torch.nn.Module):

    def __init__(self, input_scan_dim, output_dim):
        super(RwkvModel, self).__init__()
        self.input_scan_dim = input_scan_dim

        tmp = types.SimpleNamespace()
        tmp.n_layer = 4
        tmp.n_embd = 64
        tmp.head_size_a = 64 # don't change
        tmp.head_size_divisor = 8 # don't change
        self.gpt_config = tmp

        self.encoder = torch.nn.Linear(input_scan_dim, tmp.n_embd)
        self.rwkv = RWKV(self.gpt_config)
        self.readout = torch.nn.Linear(tmp.n_embd, output_dim)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        batch_size = x.size(0)
        state = torch.zeros(batch_size, self.gpt_config.n_layer * (2+self.gpt_config.head_size_a), self.gpt_config.n_embd).to(x.device)
        
        # print("1", x.shape)
        for input_t in x.squeeze(1).split(1, dim=1):
            out = self.encoder(input_t.squeeze(1))
            out, state = self.rwkv(out, state)
        return self.readout(out)

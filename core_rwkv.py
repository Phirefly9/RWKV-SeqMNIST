#  type: ignore
#  flake8: noqa
#  pylint: skip-file
#  ruff: noqa
"""
RWKV "x60a" model - does not require custom CUDA kernel to train :)

References:
https://github.com/BlinkDL/RWKV-LM

modification of RWKV v6 rnn mode to support batches of data to support
training using the rnn mode
"""


import gc
import math
import types

import torch
from torch import nn
from torch.nn import functional as F


class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            # original code has an unsqeeze when unpacking weights, I just apply it here
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def forward(self, x, state):
        new_state = state.clone()
        B = x.shape[0]
        H = self.n_head
        S = self.head_size
        i = self.layer_id
        i1 = (2+S)*self.layer_id+1
        sx = state[:, i1, :] - x
        new_state[:, i1, :] = x
        xxx = x + sx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(5, B, -1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (self.time_maa_w + mw)
        xk = x + sx * (self.time_maa_k + mk)
        xv = x + sx * (self.time_maa_v + mv)
        xr = x + sx * (self.time_maa_r + mr)
        xg = x + sx * (self.time_maa_g + mg)

        w = (self.time_decay + (torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).float()).view(B, H, S, 1)
        w = torch.exp(-torch.exp(w.float()))

        r = self.receptance(xr).view(B, H, 1, S)
        k = self.key(xk).view(B, H, S, 1) #
        v = self.value(xv).view(B, H, 1, S) #
        g = F.silu(self.gate(xg))

        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].reshape(B, H, S, S)

        a = k @ v
        x = r @ (self.time_faaaa * a + s)
        s = a + w * s

        new_state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.reshape(B, S, -1)
        x = x.flatten(1)
        x = self.ln_x(x) * g.squeeze(0)
        return self.output(x), new_state


class RWKV_Tmix_x060c(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(args.n_layer - 1, 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            # original code has an unsqeeze when unpacking weights, I just apply it here
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        # self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

    def forward(self, x, state):
        new_state = state.clone()
        B = x.shape[0]
        H = self.n_head
        S = self.head_size
        i = self.layer_id
        i1 = (2+S)*self.layer_id+1
        sx = state[:, i1, :] - x
        new_state[:, i1, :] = x
        xxx = x + sx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, -1)
        
        mr, mk, mv, mw = xxx.unbind(dim=0)
        xr = x + sx * (self.time_maa_r + mr)
        xk = x + sx * (self.time_maa_k + mk)
        xv = x + sx * (self.time_maa_v + mv)
        xw = x + sx * (self.time_maa_w + mw)
        

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        w = self.time_decay + (torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)
        w = torch.exp(-torch.exp(w.float()))


        k = k * (1-(-w.exp()).exp())

        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].reshape(B, H, S, S)

        r = r.view(B, H, 1, S)
        k = k.view(B, H, S, 1)
        v = v.view(B, H, 1, S)
        w = w.view(B, H, S, 1)

        # core rwkv kernel
        a = k @ v
        x = r @ (self.time_faaaa * a + s)
        s = a + w * s

        # after core
        new_state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.reshape(B, S, -1)
        x = x.flatten(1)
        x = self.ln_x(x)
        return self.output(x), new_state

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.head_size = args.head_size_a

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x, state):
        new_state = state.clone()
        i0 = (2+self.head_size)*self.layer_id+0
        sx = state[:, i0] - x
        xk = x + sx * self.time_maa_k
        xr = x + sx * self.time_maa_r
        new_state[:, i0] = x
        r = torch.sigmoid(self.receptance(xr))
        k = torch.square(torch.relu(self.key(xk))) # square relu, primer paper
        return r * (self.value(k)), new_state

class Block(nn.Module):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060c(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

    def forward(self, x, state):

        if self.layer_id == 0:
            x = self.ln0(x)

        tmp_out, state = self.att.forward(self.ln1(x), state)
        x = x + tmp_out
        tmp_out, state = self.ffn.forward(self.ln2(x), state)
        x = x + tmp_out

        return x, state

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)

        self.init_params() # !!! When you train RWKV from scratch, try my initialization for best performance !!!

        # print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def init_params(self):
        m = self.state_dict()
        n_params = 0

        for n in self.state_dict():
            p = m[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            # print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or n.endswith(("_w", "_w1", "_w2", "_bias")):
                if "ln_x.weight" in n:
                    layer_scale = (1+int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                # print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale) # !!! If you are using positional embedding, maybe it's better to remove block.0.ln0, and use default initialization for emb.weight instead of my uniform_(a=-1e-4, b=1e-4) !!!
                # print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd) if self.args.vocab_size > self.args.n_embd else 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                # print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight") # should always be true

                for kk in [".att.output.", ".ffn.value.", ".ffn.receptance."]:
                    if kk in n:
                        scale = 0
                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                # print(f" [scale {scale}]")

                m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                if scale == 0:
                    nn.init.zeros_(m[n])
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            n_params += m[n].numel()

        # print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x, state):

        for block in self.blocks:
            x, state = block(x, state)

        x = self.ln_out(x)

        return x, state

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())



if __name__ == "__main__":
    tmp = types.SimpleNamespace()
    tmp.n_layer = 12
    tmp.n_embd = 64
    tmp.head_size_a = 64 # don't change
    tmp.head_size_divisor = 8 # don't change

    print(tmp)
    tmp2 = RWKV(tmp)

    x = torch.rand((2, tmp.n_embd))
    state = torch.zeros(2, tmp.n_layer * (2+tmp.head_size_a), tmp.n_embd)

    out = tmp2.forward(x, state)

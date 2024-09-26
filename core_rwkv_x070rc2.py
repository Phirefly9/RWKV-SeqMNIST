"""
RWKV "x70rc2" model

implements naive recurrent rwkv x70rc2, with batch support assuming outside padding
"""


import gc
import math
import types

import torch
from torch import nn
from torch.nn import functional as F


class RWKV_Tmix_x070rc2(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ddd = torch.zeros(1, 1, args.n_embd)
            self.time_maa_x = nn.Parameter(ddd)
            self.time_maa_r = nn.Parameter(ddd)
            self.time_maa_w = nn.Parameter(ddd)
            self.time_maa_k = nn.Parameter(ddd)
            self.time_maa_v = nn.Parameter(ddd)
            self.time_maa_a = nn.Parameter(ddd)
            self.time_maa_g = nn.Parameter(ddd)

            decay_speed = torch.zeros(args.dim_att)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            self.time_faaaa = nn.Parameter(torch.zeros(self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,args.dim_att))

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*6))
            self.time_maa_w2 = nn.Parameter(torch.zeros(6, D_MIX_LORA, args.n_embd)).uniform_(-0.01, 0.01)

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att)).uniform_(-0.01, 0.01)

            D_AAA_LORA = 64
            self.time_aaa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(torch.zeros(D_AAA_LORA, args.dim_att)).uniform_(-0.01, 0.01)

            D_KKK_LORA = 64
            self.time_kkk_w1 = nn.Parameter(torch.zeros(args.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(torch.zeros(D_KKK_LORA, args.dim_att)).uniform_(-0.01, 0.01)

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(torch.zeros(args.n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(torch.zeros(D_GATE_LORA, args.dim_att)).uniform_(-0.01, 0.01)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def forward(self, x, state, new_state):
        # new_state = state.clone()
        B, T, C = x.size()
        H = self.n_head
        S = self.head_size
        i = self.layer_id
        i1 = (2+S)*self.layer_id+1
        
        xx = torch.cat((state[:, i1, :].unsqueeze(1), x[:, :-1, :]), dim=1) - x
        new_state[:, i1, :] = x[:, -1, :]
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 6, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, -1)
        mr, mw, mk, mv, ma, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xa = x + xx * (self.time_maa_a + ma)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        w = -F.softplus(-(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        g = torch.tanh(xg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk, dim=-1, p=2.0)
        a = torch.sigmoid( self.time_aaaaa + (xa @ self.time_aaa_w1) @ self.time_aaa_w2 ) * 2.0 # a is "in-context learning rate"

        k = k * torch.clamp(w*0.5,max=0).exp()

        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].reshape(B, H, S, S).float()

        # rwkv7rc2 kernel
        a = -kk
        b = kk*a
        r = r.view(B, T, H, S).float()
        k = k.view(B, T, H, S).float()
        v = v.view(B, T, H, S).float()
        a = a.view(B, T, H, S).float()
        b = b.view(B, T, H, S).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, S).float()))

        ret = []
        for t in range(T):
            kk = k[:, t, :]
            rr = r[:, t, :]
            vv = v[:, t, :]
            aa = a[:, t, :]
            bb = b[:, t, :]

            sab = torch.einsum('bhik,bhk,bhj->bhij', s, aa, bb)
            s = s * w[:, t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
            ret.append(torch.einsum('bhj,bhij->bhi', rr, s))

        x = torch.stack(ret, dim=1)
        # end kernel

        new_state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.reshape(B, S, -1)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1)*k.view(B,T, H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        return self.output(x * g.squeeze(0)), new_state

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

    def forward(self, x, state, new_state):
        # new_state = state.clone()
        i0 = (2+self.head_size)*self.layer_id+0
        sx = torch.cat((state[:, i0, :].unsqueeze(1), x[:, :-1, :]), dim=1) - x
        xk = x + sx * self.time_maa_k
        xr = x + sx * self.time_maa_r
        new_state[:, i0] = x[:, -1, :]
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

        self.att = RWKV_Tmix_x070rc2(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

    def forward(self, x, state, new_state):

        if self.layer_id == 0:
            x = self.ln0(x)

        tmp_out, new_state = self.att.forward(self.ln1(x), state, new_state)
        x = x + tmp_out
        tmp_out, new_state = self.ffn.forward(self.ln2(x), state, new_state)
        x = x + tmp_out

        return x, new_state

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

        # this is rwkv6 init, probably not correct for rwkv7
        self.init_params()

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
        new_state = torch.zeros_like(state)

        for block in self.blocks:
            x, new_state = block(x, state, new_state)

        x = self.ln_out(x)

        return x, new_state

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())



if __name__ == "__main__":
    tmp = types.SimpleNamespace()
    tmp.n_layer = 1
    tmp.n_embd = 128
    tmp.head_size_a = 64 # don't change
    tmp.head_size_divisor = 8 # don't change

    print(tmp)
    tmp2 = RWKV(tmp)

    x = torch.rand((2, 1, tmp.n_embd))
    state = torch.zeros(2, tmp.n_layer * (2+tmp.head_size_a), tmp.n_embd)

    out = tmp2.forward(x, state)

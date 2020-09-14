import torch
from torch import nn
from utils import get_logger
from transformers import PhobertModel

logger = get_logger('Model')


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=5, eps=1e-8, hidden_dim=128):
        super(SlotAttention, self).__init__()
        self._num_slots = num_slots
        self._iters = iters
        self._eps = eps
        self._scale = dim ** -0.5

        self._slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self._slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self._to_q = nn.Linear(dim, dim)
        self._to_k = nn.Linear(dim, dim)
        self._to_v = nn.Linear(dim, dim)

        self._gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)

        self._mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self._norm_input = nn.LayerNorm(dim)
        self._norm_slots = nn.LayerNorm(dim)
        self._norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self._num_slots

        mu = self._slots_mu.expand(b, n_s, -1)
        sigma = self._slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self._norm_input(inputs)
        k, v = self._to_k(inputs), self._to_v(inputs)

        for _ in range(self._iters):
            slots_prev = slots
            slots = self._norm_slots(slots)
            q = self._to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self._scale
            attn = dots.softmax(dim=1) + self._eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self._gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self._mlp(self._norm_pre_ff(slots))

        return slots


class Model(nn.Module):
    def __init__(self, num_aspect, num_polarity, embedding_dim=768, slot_attention_hidden_dim=512):
        super(Model, self).__init__()
        self.phobert = PhobertModel.from_pretrained('vinai/phobert-base')
        for param in self.phobert.parameters():
            param.requires_grad = True

        self.slot_attn = SlotAttention(num_slots=num_aspect,
                                       dim=embedding_dim,
                                       hidden_dim=slot_attention_hidden_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_polarity)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_mask):
        x = self.phobert(x, attention_mask=attn_mask)
        x = self.slot_attn(x[0])
        x = self.linear(x)
        x = self.softmax(x)
        return x

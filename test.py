import os
import click
import torch
from model import Model


@click.option('--model_path', type=str, default='outputs/checkpoint/cpkt.vndee', help='Path to model checkpoint')
@click.option('--device', type=str, default='cuda', help='Test device')
def test(model_path: str,
         device: str):
    model = Model(num_aspect=56, num_polarity=4).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    print(model)


if __name__ == '__main__':
    test()
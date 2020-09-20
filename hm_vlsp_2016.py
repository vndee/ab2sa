import os
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from utils import get_logger
from vncorenlp import VnCoreNLP
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from compute_word_relevent_polarity import word_in_doc, term_frequencies

logger = get_logger('HybridDataset')


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_size=768 + 3, lstm_hidden_size=512, num_classes=4, device='cuda'):
        super(BiLSTM_Attention, self).__init__()
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        self.device = device

        self.lstm = torch.nn.LSTM(self.embedding_size, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.out = torch.nn.Linear(self.embedding_size, self.num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.lstm_hidden_size * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = torch.nn.functional.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data

    def forward(self, x, wrp):
        x = torch.cat((x, wrp), dim=-1)
        hidden_state = Variable(torch.zeros(1 * 2, x.shape[0], self.lstm_hidden_size, device=self.device))
        cell_state = Variable(torch.zeros(1 * 2, x.shape[0], self.lstm_hidden_size, device=self.device))

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
        attn_output, attention = self.attention_net(output, final_hidden_state)

        return x


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Lan cuoi cung')
    argument_parser.add_argument('--data', type=str, default='Hotel')
    argument_parser.add_argument('--device', type=str, default='cuda')
    argument_parser.add_argument('--batch_size', type=int, default=2)
    args = argument_parser.parse_args()

    train, test = HybridDataset(data=args.data, file='train'), HybridDataset(data=args.data, file='test')

    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test, shuffle=True, batch_size=args.batch_size)

    net = BiLSTM_Attention().to(args.device)
    phobert = AutoModel.from_pretrained('vinai/phobert-base').to(args.device)

    logger.info(phobert)
    logger.info(net)

    for idx, (items, wrp, labels) in enumerate(train_loader):
        items = items.to(args.device)
        wrp = wrp.to(args.device)
        labels = labels.to(args.device)
        attn_mask = (items > 0).to(args.device)

        inputs = phobert(items, attention_mask=attn_mask)[0]
        preds= net(inputs, wrp)

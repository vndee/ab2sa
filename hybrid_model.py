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


def padding(x, max_length):
    cls_id, eos_id, pad_id = 0, 0, 1
    temp = torch.zeros(max_length, dtype=torch.long)
    if x.shape[0] > max_length:
        x = x[: max_length]
    temp[0: x.shape[0]] = x
    temp[-1] = eos_id
    return temp


class HybridDataset(Dataset):
    def __init__(self,
                 data='Hotel',
                 file='train',
                 path=os.path.join('dataset', 'VLSP2018'),
                 max_length=256):
        super(HybridDataset, self).__init__()
        self.max_length = max_length
        with open(os.path.join(path, f'VLSP2018-SA-{data}-{file}.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.file = stream.read()

        self.data = data.lower()

        self.aspect_hotel = ['rooms#prices', 'room_amenities#general', 'room_amenities#prices', 'hotel#prices',
                             'rooms#cleanliness', 'location#general', 'facilities#quality', 'facilities#miscellaneous',
                             'hotel#design&features', 'facilities#general', 'food&drinks#style&options', 'hotel#miscellaneous',
                             'food&drinks#quality', 'rooms#miscellaneous', 'rooms#design&features', 'hotel#comfort',
                             'food&drinks#prices', 'hotel#cleanliness', 'room_amenities#comfort', 'rooms#general',
                             'room_amenities#quality', 'rooms#quality', 'facilities#design&features', 'facilities#cleanliness',
                             'food&drinks#miscellaneous', 'room_amenities#miscellaneous', 'hotel#general', 'service#general',
                             'rooms#comfort', 'room_amenities#cleanliness', 'facilities#comfort', 'facilities#prices',
                             'room_amenities#design&features', 'hotel#quality']

        self.aspect_restaurant = ['drinks#quality', 'drinks#style&options', 'service#general', 'restaurant#prices',
                                  'food#quality', 'drinks#prices', 'ambience#general', 'food#prices', 'restaurant#miscellaneous',
                                  'restaurant#general', 'location#general', 'food#style&options']

        self.num_aspect = 1 + self.aspect_hotel.__len__() if data == 'Hotel' else self.aspect_restaurant.__len__()
        self.polarities = ['negative', 'neural', 'positive']
        self.num_polarity = 1 + self.polarities.__len__()
        self.file = self.file.split('\n\n')

        self.rdr_segmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

        self.doc = ['NEG', 'NEU', 'POS']

    def label_encode(self, x):
        lb_list = list()
        for item in x.split('},'):
            item = item.replace('{', '').replace('}', '')
            aspect, polarity = item.split(',')

            if self.data == 'hotel':
                asp = self.aspect_hotel.index(aspect.strip().lower())
            elif self.data == 'restaurant':
                asp = self.aspect_restaurant.index(aspect.strip().lower())
            else:
                asp = None

            plr = ['negative', 'neutral', 'positive'].index(polarity.strip())
            lb_list.append((asp, plr))

        return lb_list

    def __getitem__(self, item):
        lines = self.file[item].split('\n')
        label = self.label_encode(lines[2].strip())

        labels = torch.ones((self.num_aspect, 1)) * 3
        for lb in label:
            labels[lb[0]] = lb[1]

        text = lines[1].strip()
        text = self.rdr_segmenter.tokenize(text)

        text = [item for sublist in text for item in sublist]
        word_respect_to_polarity = None
        for w in text:
            if w in term_frequencies:
                term_frequency = term_frequencies[w]
            else:
                term_frequency = {
                    'NEG': 0,
                    'NEU': 0,
                    'POS': 0
                }

            lst = [v for k, v in term_frequency.items()]
            lst = torch.nn.functional.softmax(torch.from_numpy(np.asarray(lst)).type(torch.FloatTensor), dim=0)

            if word_respect_to_polarity is None:
                word_respect_to_polarity = lst
            elif len(word_respect_to_polarity.shape) == 1:
                word_respect_to_polarity = torch.stack((word_respect_to_polarity, lst), dim=0)
            else:
                word_respect_to_polarity = torch.cat((word_respect_to_polarity, lst.unsqueeze(0)), dim=0)

        text = ' '.join(text)
        text = torch.tensor(self.tokenizer.encode(text, padding=self.max_length, truncation=True))
        labels = labels.squeeze(-1).type(torch.LongTensor)
        labels = torch.nn.functional.one_hot(labels)

        if word_respect_to_polarity.shape[0] < self.max_length:
            adjust_size = self.max_length - word_respect_to_polarity.shape[0]
            adjust_tensor = torch.zeros((adjust_size, 3))
            word_respect_to_polarity = torch.cat((word_respect_to_polarity, adjust_tensor), dim=0)

        return padding(text, self.max_length), word_respect_to_polarity, labels

    def __len__(self):
        return self.file.__len__()


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_size=768 + 3, lstm_hidden_size=512, num_classes=4, num_aspect=10, device='cuda'):
        super(BiLSTM_Attention, self).__init__()
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        self.num_aspect = num_aspect
        self.device = device

        self.lstm = torch.nn.LSTM(self.embedding_size, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.out = torch.nn.Linear(2 * self.lstm_hidden_size, self.num_aspect * self.num_classes)

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
        x = self.out(attn_output)
        return x


def calc_loss(preds, targets):
    return 0


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
        preds = net(inputs, wrp)

        loss = calc_loss(preds, labels)

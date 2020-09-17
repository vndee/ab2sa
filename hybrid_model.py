import os
import math
import torch
from utils import get_logger
from vncorenlp import VnCoreNLP
from transformers import PhobertTokenizer
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
        self.tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')

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

        word_respect_to_polarity = list()
        for w in text[0]:
            if w in term_frequencies:
                term_frequency = term_frequencies[w]
            else:
                term_frequency = {
                    'NEG': 0,
                    'NEU': 0,
                    'POS': 0
                }

            if w in word_in_doc:
                w_in_doc = word_in_doc[w]
            else:
                w_in_doc = {}

            tf_idf = []
            for doc in self.doc:
                tf = 1 + math.log(term_frequency[doc]) if term_frequency[doc] != 0 else 1
                idf = math.log(3 / (1 + len(w_in_doc)), 10)
                tf_idf.append(tf * idf)

            word_respect_to_polarity.append(tf_idf)

        text = ' '.join(text[0])
        text = torch.tensor(self.tokenizer.encode(text))
        labels = labels.squeeze(-1).type(torch.LongTensor)
        labels = torch.nn.functional.one_hot(labels)
        return padding(text, self.max_length), word_respect_to_polarity, labels

    def __len__(self):
        return self.file.__len__()


if __name__ == '__main__':
    dataset = HybridDataset()
    for item, wrp, lb in dataset:
        print(wrp)

import os
import torch
from utils import get_logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

logger = get_logger('Data Loader')


def padding(x, max_length):
    cls_id, eos_id, pad_id = 0, 0, 1
    temp = torch.zeros(max_length, dtype=torch.long)
    if x.shape[0] > max_length:
        x = x[: max_length]
    temp[0: x.shape[0]] = x
    temp[-1] = eos_id
    return temp


class VLSP2018BertPair(Dataset):
    def __init__(self,
                 data='Hotel',
                 file='train',
                 path=os.path.join('dataset', 'VLSP2018'),
                 max_length=256):
        super(VLSP2018BertPair, self).__init__()
        self.max_length = max_length
        with open(os.path.join(path, f'VLSP2018-SA-{data}-{file}.prod'), mode='r', encoding='utf-8-sig') as stream:
            self.file = stream.read()

        self.data = data.lower()

        # self.entity_hotel = ['HOTEL', 'ROOMS', 'ROOM_AMENITIES', 'FACILITIES', 'SERVICE', 'LOCATION', 'FOOD&DRINKS']
        # self.attribute_hotel = ['GENERAL', 'PRICES', 'DESIGN&FEATURES', 'CLEANLINESS', 'COMFORT', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_hotel = [f'{x}#{y}' for x in self.entity_hotel for y in self.attribute_hotel]
        self.aspect_hotel = ['rooms#prices', 'room_amenities#general', 'room_amenities#prices', 'hotel#prices',
                             'rooms#cleanliness', 'location#general', 'facilities#quality', 'facilities#miscellaneous',
                             'hotel#design&features', 'facilities#general', 'food&drinks#style&options', 'hotel#miscellaneous',
                             'food&drinks#quality', 'rooms#miscellaneous', 'rooms#design&features', 'hotel#comfort',
                             'food&drinks#prices', 'hotel#cleanliness', 'room_amenities#comfort', 'rooms#general',
                             'room_amenities#quality', 'rooms#quality', 'facilities#design&features', 'facilities#cleanliness',
                             'food&drinks#miscellaneous', 'room_amenities#miscellaneous', 'hotel#general', 'service#general',
                             'rooms#comfort', 'room_amenities#cleanliness', 'facilities#comfort', 'facilities#prices',
                             'room_amenities#design&features', 'hotel#quality']

        # self.entity_restaurant = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
        # self.attribute_restaurant = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_restaurant = [f'{x}#{y}' for x in self.entity_restaurant for y in self.attribute_restaurant]
        self.aspect_restaurant = ['drinks#quality', 'drinks#style&options', 'service#general', 'restaurant#prices',
                                  'food#quality', 'drinks#prices', 'ambience#general', 'food#prices', 'restaurant#miscellaneous',
                                  'restaurant#general', 'location#general', 'food#style&options']

        self.polarities = ['negative', 'neural', 'positive']

        self.file = self.file.strip().split('\n\n')
        self.rdr_segmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    def label_encode(self, x):
        x = x.split('\n')

        aspect, polarity = x[0].split(',')
        lb = None

        if self.data == 'hotel':
            lb = self.aspect_hotel.index(aspect.lower())
        elif self.data == 'restaurant':
            lb = self.aspect_restaurant.index(aspect.lower())

        polarity = polarity.strip()
        polarity = ['negative', 'neutral', 'positive'].index(polarity)
        aspect = aspect.replace('#', ', ').replace('&', ' and ').lower()
        return aspect, lb, polarity

    def __getitem__(self, item):
        lines = self.file[item].split('\n')

        try:
            label = self.label_encode(lines[1].strip())
        except Exception as ex:
            logger.info(lines)
            logger.exception(ex)
            return None, None

        text = f'{lines[0].strip()} {label[0]}'
        text = self.rdr_segmenter.tokenize(text)
        text = ' '.join(text[0])
        text = torch.tensor(self.tokenizer.encode(text))
        return padding(text, self.max_length), label[-1]

    def __len__(self):
        return self.file.__len__()


class VLSP2018(Dataset):
    def __init__(self,
                 data='Hotel',
                 file='train',
                 path=os.path.join('dataset', 'VLSP2018'),
                 max_length=256):
        super(VLSP2018, self).__init__()
        self.max_length = max_length
        with open(os.path.join(path, f'VLSP2018-SA-{data}-{file}.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.file = stream.read()

        self.data = data.lower()

        # self.entity_hotel = ['HOTEL', 'ROOMS', 'ROOM_AMENITIES', 'FACILITIES', 'SERVICE', 'LOCATION', 'FOOD&DRINKS']
        # self.attribute_hotel = ['GENERAL', 'PRICES', 'DESIGN&FEATURES', 'CLEANLINESS', 'COMFORT', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_hotel = [f'{x}#{y}' for x in self.entity_hotel for y in self.attribute_hotel]
        self.aspect_hotel = ['rooms#prices', 'room_amenities#general', 'room_amenities#prices', 'hotel#prices',
                             'rooms#cleanliness', 'location#general', 'facilities#quality', 'facilities#miscellaneous',
                             'hotel#design&features', 'facilities#general', 'food&drinks#style&options', 'hotel#miscellaneous',
                             'food&drinks#quality', 'rooms#miscellaneous', 'rooms#design&features', 'hotel#comfort',
                             'food&drinks#prices', 'hotel#cleanliness', 'room_amenities#comfort', 'rooms#general',
                             'room_amenities#quality', 'rooms#quality', 'facilities#design&features', 'facilities#cleanliness',
                             'food&drinks#miscellaneous', 'room_amenities#miscellaneous', 'hotel#general', 'service#general',
                             'rooms#comfort', 'room_amenities#cleanliness', 'facilities#comfort', 'facilities#prices',
                             'room_amenities#design&features', 'hotel#quality']

        # self.entity_restaurant = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
        # self.attribute_restaurant = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_restaurant = [f'{x}#{y}' for x in self.entity_restaurant for y in self.attribute_restaurant]
        self.aspect_restaurant = ['drinks#quality', 'drinks#style&options', 'service#general', 'restaurant#prices',
                                  'food#quality', 'drinks#prices', 'ambience#general', 'food#prices', 'restaurant#miscellaneous',
                                  'restaurant#general', 'location#general', 'food#style&options']

        self.num_aspect = 1 + self.aspect_hotel.__len__() if data == 'Hotel' else self.aspect_restaurant.__len__()
        self.polarities = ['negative', 'neural', 'positive']
        self.num_polarity = 1 + self.polarities.__len__()
        self.file = self.file.split('\n\n')

        self.rdr_segmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

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
            # aspect = aspect.replace('#', ', ').replace('&', ' and ').lower()

        return lb_list

    def __getitem__(self, item):
        lines = self.file[item].split('\n')
        label = self.label_encode(lines[2].strip())

        labels = torch.ones((self.num_aspect, 1)) * 3
        for lb in label:
            labels[lb[0]] = lb[1]

        text = lines[1].strip()
        text = self.rdr_segmenter.tokenize(text)
        text = ' '.join([' '.join(tex) for tex in text])
        text = torch.tensor(self.tokenizer.encode(text, max_length=self.max_length))
        labels = labels.squeeze(-1).type(torch.LongTensor)
        labels = torch.nn.functional.one_hot(labels)
        return padding(text, self.max_length), labels

    def __len__(self):
        return self.file.__len__()


class VLSP2018BertPairTopic(Dataset):
    def __init__(self,
                 data='Hotel',
                 file='train',
                 path=os.path.join('dataset', 'VLSP2018'),
                 max_length=256):
        super(VLSP2018BertPairTopic, self).__init__()
        self.max_length = max_length
        with open(os.path.join(path, f'VLSP2018-SA-{data}-{file}.prod'), mode='r', encoding='utf-8-sig') as stream:
            self.file = stream.read()

        self.data = data.lower()

        # self.entity_hotel = ['HOTEL', 'ROOMS', 'ROOM_AMENITIES', 'FACILITIES', 'SERVICE', 'LOCATION', 'FOOD&DRINKS']
        # self.attribute_hotel = ['GENERAL', 'PRICES', 'DESIGN&FEATURES', 'CLEANLINESS', 'COMFORT', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_hotel = [f'{x}#{y}' for x in self.entity_hotel for y in self.attribute_hotel]

        self.aspect_hotel = ['rooms#prices', 'room_amenities#general', 'room_amenities#prices', 'hotel#prices',
                             'rooms#cleanliness', 'location#general', 'facilities#quality', 'facilities#miscellaneous',
                             'hotel#design&features', 'facilities#general', 'food&drinks#style&options', 'hotel#miscellaneous',
                             'food&drinks#quality', 'rooms#miscellaneous', 'rooms#design&features', 'hotel#comfort',
                             'food&drinks#prices', 'hotel#cleanliness', 'room_amenities#comfort', 'rooms#general',
                             'room_amenities#quality', 'rooms#quality', 'facilities#design&features', 'facilities#cleanliness',
                             'food&drinks#miscellaneous', 'room_amenities#miscellaneous', 'hotel#general', 'service#general',
                             'rooms#comfort', 'room_amenities#cleanliness', 'facilities#comfort', 'facilities#prices',
                             'room_amenities#design&features', 'hotel#quality']

        # self.entity_restaurant = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
        # self.attribute_restaurant = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_restaurant = [f'{x}#{y}' for x in self.entity_restaurant for y in self.attribute_restaurant]
        self.aspect_restaurant = ['drinks#quality', 'drinks#style&options', 'service#general', 'restaurant#prices',
                                  'food#quality', 'drinks#prices', 'ambience#general', 'food#prices', 'restaurant#miscellaneous',
                                  'restaurant#general', 'location#general', 'food#style&options']

        self.polarities = ['negative', 'neural', 'positive']

        self.file = self.file.split('\n\n')
        self.rdr_segmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    def label_encode(self, x):
        x = x.split('\n')

        aspect, polarity = x[0].split(',')
        lb = None

        if self.data == 'hotel':
            lb = self.aspect_hotel.index(aspect.strip().lower())
        elif self.data == 'restaurant':
            lb = self.aspect_restaurant.index(aspect.strip().lower())

        polarity = polarity.strip()
        polarity = ['negative', 'neutral', 'positive'].index(polarity)
        aspect = aspect.replace('#', ', ').replace('&', ' and ').lower()
        return aspect, lb, polarity

    def __getitem__(self, item):
        lines = self.file[item].split('\n')
        label = self.label_encode(lines[1].strip())

        text = f'{lines[0].strip()} {label[0]}'
        text = self.rdr_segmenter.tokenize(text)
        text = ' '.join(text[0])
        text = torch.tensor(self.tokenizer.encode(text))
        return padding(text, self.max_length), label[1]

    def __len__(self):
        return self.file.__len__()


class VLSP2018ConditionalBert(Dataset):
    def __init__(self,
                 data='Hotel',
                 file='train',
                 path=os.path.join('dataset', 'VLSP2018'),
                 max_length=256):
        super(VLSP2018ConditionalBert, self).__init__()
        self.max_length = max_length
        with open(os.path.join(path, f'VLSP2018-SA-{data}-{file}.prod'), mode='r', encoding='utf-8-sig') as stream:
            self.file = stream.read()

        self.data = data.lower()

        # self.entity_hotel = ['HOTEL', 'ROOMS', 'ROOM_AMENITIES', 'FACILITIES', 'SERVICE', 'LOCATION', 'FOOD&DRINKS']
        # self.attribute_hotel = ['GENERAL', 'PRICES', 'DESIGN&FEATURES', 'CLEANLINESS', 'COMFORT', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_hotel = [f'{x}#{y}' for x in self.entity_hotel for y in self.attribute_hotel]
        self.aspect_hotel = ['rooms#prices', 'room_amenities#general', 'room_amenities#prices', 'hotel#prices',
                             'rooms#cleanliness', 'location#general', 'facilities#quality', 'facilities#miscellaneous',
                             'hotel#design&features', 'facilities#general', 'food&drinks#style&options', 'hotel#miscellaneous',
                             'food&drinks#quality', 'rooms#miscellaneous', 'rooms#design&features', 'hotel#comfort',
                             'food&drinks#prices', 'hotel#cleanliness', 'room_amenities#comfort', 'rooms#general',
                             'room_amenities#quality', 'rooms#quality', 'facilities#design&features', 'facilities#cleanliness',
                             'food&drinks#miscellaneous', 'room_amenities#miscellaneous', 'hotel#general', 'service#general',
                             'rooms#comfort', 'room_amenities#cleanliness', 'facilities#comfort', 'facilities#prices',
                             'room_amenities#design&features', 'hotel#quality']

        # self.entity_restaurant = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
        # self.attribute_restaurant = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        # self.aspect_restaurant = [f'{x}#{y}' for x in self.entity_restaurant for y in self.attribute_restaurant]
        self.aspect_restaurant = ['drinks#quality', 'drinks#style&options', 'service#general', 'restaurant#prices',
                                  'food#quality', 'drinks#prices', 'ambience#general', 'food#prices', 'restaurant#miscellaneous',
                                  'restaurant#general', 'location#general', 'food#style&options']

        self.cnt = 0
        self.aspect_set = dict()
        self.polarities = ['negative', 'neural', 'positive']

        self.file = self.file.strip().split('\n\n')
        self.rdr_segmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    def label_encode(self, x):
        x = x.split('\n')

        aspect, polarity = x[0].split(',')
        lb, _lb = None, None

        if self.data == 'hotel':
            lb = self.aspect_hotel.index(aspect.strip().lower())
            _lb = torch.zeros((self.aspect_hotel.__len__()))
        elif self.data == 'restaurant':
            lb = self.aspect_restaurant.index(aspect.strip().lower())
            _lb = torch.zeros((self.aspect_restaurant.__len__()))

        _lb[lb] = 1

        polarity = polarity.strip()
        polarity = ['negative', 'neutral', 'positive'].index(polarity)
        aspect = aspect.replace('#', ', ').replace('&', ' and ').lower()
        return aspect, _lb, polarity

    def __getitem__(self, item):
        lines = self.file[item].split('\n')
        label = self.label_encode(lines[1].strip())

        text = f'{lines[0].strip()}'
        text = self.rdr_segmenter.tokenize(text)
        text = ' '.join(text[0])
        text = torch.tensor(self.tokenizer.encode(text))
        return padding(text, self.max_length), label[1], label[-1]

    def __len__(self):
        return self.file.__len__()


# data = VLSP2018ConditionalBert(data='Restaurant')
# for item, lb, pl in data:
#     print(item.shape, lb.shape, pl)

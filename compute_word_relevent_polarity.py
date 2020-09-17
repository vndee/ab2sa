import os
from utils import get_logger
from vncorenlp import VnCoreNLP
from transformers import PhobertTokenizer

logger = get_logger('Compute Word Relevant Polarity')


rdr_segmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')
term_frequencies = dict()
word_in_doc = dict()

with open(os.path.join('dataset', 'VLSP2016', 'SA-2016.train'), 'r') as stream:
    corpus = stream.read().split('\n')
    for line in corpus:
        if line.strip() == '':
            continue

        line = line.split('\t')
        content, document = line[0], line[1]
        content = rdr_segmenter.tokenize(content)[0]

        for word in content:
            term_frequencies[word] = {
                'NEG': 0,
                'NEU': 0,
                'POS': 0
            }
            term_frequencies[word][document] += 1

            if word not in word_in_doc:
                word_in_doc[word] = dict()
                word_in_doc[word] = {
                    document: 1
                }
            else:
                word_in_doc[word][document] = 1

logger.info('Computed accessories.')
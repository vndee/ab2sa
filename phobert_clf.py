import torch
import torch.nn as nn
from utils import get_logger
from transformers import PhobertTokenizer, PhobertModel, RobertaConfig, RobertaForSequenceClassification

logger = get_logger('PhobertForSequenceClassification')


class PhobertABSA(nn.Module):
    def __init__(self, num_labels=4):
        super(PhobertABSA, self).__init__()
        self.phobert = PhobertModel.from_pretrained('vinai/phobert-base')
        self.linear_1 = nn.Linear(768, 768, bias=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.linear_2 = nn.Linear(768, num_labels, bias=True)

    def forward(self, x, attn_mask):
        outputs = self.phobert(x, attention_mask=attn_mask)
        cls = outputs[0]
        x = self.linear_1(cls)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


if __name__ == '__main__':
    clf = PhobertABSA()
    logger.info(clf)

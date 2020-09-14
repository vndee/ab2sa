import torch
import numpy as np
import torch.nn as nn
from utils import get_logger
from tqdm import tqdm
from data import VLSP2018BertPairTopic
from train import evaluate
from torch.utils.data import DataLoader
from model import SlotAttention
from transformers import PhobertTokenizer, PhobertModel, RobertaConfig, RobertaForSequenceClassification


logger = get_logger('PhobertForSequenceClassification')


class PhobertABSA(nn.Module):
    def __init__(self, num_labels=3):
        super(PhobertABSA, self).__init__()
        self.phobert = PhobertModel.from_pretrained('vinai/phobert-base')
        self.slot_attn = SlotAttention(num_slots=3,
                                       dim=768,
                                       hidden_dim=768)
        self.linear_1 = nn.Linear(768, 1, bias=True)

    def forward(self, x, attn_mask):
        outputs = self.phobert(x, attention_mask=attn_mask)
        x = self.slot_attn(outputs[0])
        # cls = outputs[0][:, 0, :]
        x = self.linear_1(x)
        x = x.squeeze(-1)
        logger.info(x.shape)
        return x


if __name__ == '__main__':
    num_epochs = 10
    device = 'cuda'
    batch_size = 2
    accumulation_step = 50


    train, test = VLSP2018BertPairTopic(data='Hotel', file='train'), VLSP2018BertPairTopic(data='Hotel', file='test')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    num_lb = train.aspect_hotel.__len__()
    clf = PhobertABSA(num_lb).to(device)
    logger.info(clf)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(clf.parameters(), lr=3e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              (train.__len__() // batch_size) * num_epochs,
                                                              eta_min=0)

    for epoch in range(num_epochs):
        train_loss, test_loss = None, None
        train_acc, test_acc = None, None
        train_f1, test_f1 = None, None
        _preds, _targets = None, None

        for idx, (items, labels) in enumerate(tqdm(train_loader, desc='Training')):
            items = items.to(device)
            attn_masks = (items > 0).to(device)
            labels = labels.to(device)

            preds = clf(items, attn_masks)
            loss = criterion(preds, labels)

            loss.backward()
            if idx != 0 and idx % accumulation_step == 0:
                optimizer.step()
                lr_scheduler.step()

            train_loss = train_loss + loss.item() if train_loss is not None else loss.item()

            preds = torch.argmax(preds, dim=-1).view(-1)
            labels = labels.view(-1)
            # labels = torch.argmax(labels, dim=-1).view(-1)

            if device == 'cuda':
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            else:
                preds = preds.detach().numpy()
                labels = labels.detach().numpy()

            _preds = np.atleast_1d(preds) if _preds is None else np.concatenate([_preds, np.atleast_1d(preds)])
            _targets = np.atleast_1d(labels) if _targets is None else np.concatenate([_targets, np.atleast_1d(labels)])

        train_acc, train_f1 = evaluate(_preds, _targets)

        with torch.no_grad():
            _preds, _targets = None, None
            for idx, (items, labels) in enumerate(tqdm(test_loader, desc='Evaluation')):
                items = items.to(device)
                attn_masks = (items > 0).to(device)
                labels = labels.to(device)

                preds = clf(items, attn_masks)

                loss = criterion(preds, labels)
                test_loss = test_loss + loss.item() if test_loss is not None else loss.item()

                preds = torch.argmax(preds, dim=-1).view(-1)
                labels = labels.view(-1)
                # labels = torch.argmax(labels, dim=-1).view(-1)

                if device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    labels = labels.detach().numpy()

                _preds = np.atleast_1d(preds) if _preds is None else np.concatenate([_preds, np.atleast_1d(preds)])
                _targets = np.atleast_1d(labels) if _targets is None else np.concatenate(
                    [_targets, np.atleast_1d(labels)])

            test_acc, test_f1 = evaluate(_preds, _targets)

        logger.info(f'[{epoch + 1}/{num_epochs}] Training loss: {train_loss} | Train Accuracy: {train_acc} | Train F1: '
                    f'{train_f1} | Evaluation loss: {test_loss} | Evaluation Accuracy: {test_acc} | Test F1: {test_f1}')

    logger.info(clf)

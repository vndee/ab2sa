import os
import click
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from data import VLSP2018
from utils import get_logger
from model import Model
from loss import FocalLoss

logger = get_logger('Trainer')


def evaluate(_preds, _targets):
    report = classification_report(_preds,
                                   _targets,
                                   output_dict=True,
                                   zero_division=1)

    acc = report['accuracy']
    f1 = report['macro avg']['f1-score']

    return acc, f1


@click.command(name='AB2SA Trainer')
@click.option('--data', type=str, default='Hotel', help='Dataset use to train')
@click.option('--device', type=str, default='cuda', help='Device use to train')
@click.option('--gpus', type=str, default='0', help='GPUs id')
@click.option('--batch_size', type=int, default=2, help='Training batch size')
@click.option('--num_epochs', type=int, default=10, help='Number of training epoch')
@click.option('--learning_rate', type=float, default=2e-3, help='Learning rate')
@click.option('--accumulation_step', type=int, default=100, help='Optimizer accumulation step')
@click.option('--experiment_path', type=str, default='outputs/', help='Experiment output path')
def train(data: str,
          device: str,
          gpus: str,
          batch_size: int,
          num_epochs: int,
          learning_rate: float,
          accumulation_step: int,
          experiment_path: str) -> None:
    # Set environment variable for specific GPU training
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    os.makedirs(experiment_path, exist_ok=True)

    # Load dataset
    train_dataset, test_dataset = VLSP2018(data=data, file='train'), VLSP2018(data=data, file='test')
    train_loader, test_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size),\
                                DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # Build model
    num_aspect, num_polarity = train_dataset.num_aspect, train_dataset.num_polarity
    model = Model(num_aspect=num_aspect, num_polarity=num_polarity).to(device)
    if gpus.split(',').__len__() > 1:
        model = torch.nn.DataParallel(model)

    # Criterion
    criterion = FocalLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MultiLabelMarginLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              (train_dataset.__len__() // batch_size) * num_epochs,
                                                              eta_min=0)

    # Train model
    best_accuracy = 0.0
    df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss'])
    for epoch in range(num_epochs):
        train_loss = 0.0
        _preds, _targets = None, None
        train_f1, train_tt_f1 = 0, 0

        for idx, (items, labels) in enumerate(tqdm(train_loader, desc=f'Training epoch {epoch}/{num_epochs}')):
            model.train()
            items = items.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            attn_mask = (items > 0).to(device)
            preds = model(items, attn_mask)

            loss = criterion(preds, labels)
            # loss = None
            # for idx in range(num_aspect):
            #     pred = preds[:, idx, :]
            #     label = torch.argmax(labels[:, idx, :], dim=-1).type(torch.LongTensor).to(device)
            #     _loss = criterion(pred, label)
            #     loss = loss + _loss if loss is not None else _loss

            loss.backward()
            if idx != 0 and idx % accumulation_step == 0:
                optimizer.step()
                lr_scheduler.step()

            # calc accuracy
            train_loss = train_loss + loss.item()
            preds = torch.argmax(preds, dim=-1).view(-1)
            labels = torch.argmax(labels, dim=-1).view(-1)

            if device == 'cuda':
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            else:
                preds = preds.detach().numpy()
                labels = labels.detach().numpy()

            # intersect_pl, A, B = 0, 0, 0
            # intersect_ap, all_ap = 0, 0
            # for i in range(preds.shape[0]):
            #     for j in range(preds.shape[1]):
            #         if preds[i][j] < 3 and labels[i][j] < 3:
            #             intersect_ap += 1
            #         if preds[i][j] < 3:
            #             A += 1
            #         if labels[i][j] < 3:
            #             B += 1
            #         if preds[i][j] < 3 and labels[i][j] < 3 and preds[i][j] == labels[i][j]:
            #             intersect_pl += 1
            #
            # if A != 0 and B != 0:
            #     precision, recall = intersect_ap / A, intersect_ap / B
            # else:
            #     precision, recall = 0, 0
            #
            # if precision + recall > 0:
            #     train_f1 += (2 * precision * recall) / (precision + recall)
            # train_tt_f1 += preds.shape[0]

            _preds = np.atleast_1d(preds) if _preds is None else np.concatenate([_preds, np.atleast_1d(preds)])
            _targets = np.atleast_1d(labels) if _targets is None else np.concatenate([_targets, np.atleast_1d(labels)])

        train_acc, train_f1 = evaluate(_preds, _targets)

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            _preds, _targets = None, None
            test_f1, test_tt_f1 = 0, 0

            for idx, (items, labels) in enumerate(tqdm(test_loader, desc=f'Validation epoch {epoch}/{num_epochs}')):
                items = items.to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                attn_mask = (items > 0).to(device)
                preds = model(items, attn_mask)

                loss = criterion(preds, labels)
                # loss = None
                # for idx in range(num_aspect):
                #     pred = preds[:, idx, :]
                #     label = torch.argmax(labels[:, idx, :], dim=-1).type(torch.LongTensor).to(device)
                #     _loss = criterion(pred, label)
                #     loss = loss + _loss if loss is not None else _loss

                val_loss = val_loss + loss.item()
                preds = torch.argmax(preds, dim=-1).view(-1)
                labels = torch.argmax(labels, dim=-1).view(-1)

                if device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    labels = labels.detach().numpy()

                # intersect_pl, A, B = 0, 0, 0
                # intersect_ap, all_ap = 0, 0
                # for i in range(preds.shape[0]):
                #     for j in range(preds.shape[1]):
                #         if preds[i][j] < 3 and labels[i][j] < 3:
                #             intersect_ap += 1
                #         if preds[i][j] < 3:
                #             A += 1
                #         if labels[i][j] < 3:
                #             B += 1
                #         if preds[i][j] < 3 and labels[i][j] < 3 and preds[i][j] == labels[i][j]:
                #             intersect_pl += 1
                #
                # if A != 0 and B != 0:
                #     precision, recall = intersect_ap / A, intersect_ap / B
                # else:
                #     precision, recall = 0, 0
                #
                # if precision + recall > 0:
                #     test_f1 += (2 * precision * recall) / (precision + recall)
                # test_tt_f1 += preds.shape[0]

                _preds = np.atleast_1d(preds) if _preds is None else np.concatenate([_preds, np.atleast_1d(preds)])
                _targets = np.atleast_1d(labels) if _targets is None else np.concatenate([_targets, np.atleast_1d(labels)])

            val_acc, val_f1 = evaluate(_preds, _targets)
            logger.info(f'[{epoch}/{num_epochs}] train_acc: {train_acc} - train_f1: {train_f1} - train_loss: '
                        f'{train_loss} - val_acc: {val_acc} - val_f1: {val_f1} - val_loss: {val_loss}')

            if best_accuracy < val_f1:
                best_accuracy = val_f1

                if not os.path.exists(os.path.join(experiment_path, 'checkpoints')):
                    os.makedirs(os.path.join(experiment_path, 'checkpoints'))

                saved_path = os.path.join(experiment_path, 'checkpoints', 'cpkt.vndee')
                torch.save(model.state_dict(), saved_path)

                logger.info(f'New state-of-the-art model detected. Save to {saved_path}')

    df.to_csv(f'{experiment_path}/history.csv')


if __name__ == '__main__':
    logger.info('Start Slot Attention Classifier for Aspect-based Sentiment Analysis')
    train()

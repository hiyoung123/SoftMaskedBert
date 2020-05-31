#!usr/bin/env python
#-*- coding:utf-8 -*-

import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer, BertModel, BertConfig
from optim_schedule import ScheduledOptim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import SoftMaskedBert
from sklearn.model_selection import KFold
MAX_INPUT_LEN = 512

class SoftMaskedBertTrainer():
    def __init__(self, bert, tokenizer, device, hidden=256, layer_n=1, lr=2e-5, gama=0.8, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=10000):

        self.device = device
        self.bert = bert
        self.model = SoftMaskedBert(bert, tokenizer, hidden, layer_n).to(self.device)

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, hidden, n_warmup_steps=warmup_steps)
        self.criterion_c = nn.NLLLoss()
        self.criterion_d = nn.BCELoss()
        self.gama = gama
        self.log_freq = 10

    def train(self, train_data, epoch):
        self.model.train()
        return self.iteration(epoch, train_data)

    def evaluate(self, val_data, epoch):
        self.model.eval()
        return self.iteration(epoch, val_data, train=False)

    def inference(self, test_data):
        self.model.eval()

    def save(self, file_path):
        torch.save(self.model.cpu(), file_path)
        self.model.to(self.device)
        print('Model save {}'.format(file_path))

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "val"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(data["random_text"]) #prob [batch_size, seq_len, 1]
            # label = data["label"].reshape(-1,prob.shape[1], prob.shape[-1]) #prob [batch_size, seq_len]
            prob = prob.reshape(-1, prob.shape[1])
            # prob = prob.transpose(1, 2)
            # label = data['label'].reshape(-1, prob.shape[1], prob.shape[-1])
            # p = prob.reshape(prob.shape[0]*prob.shape[1],-1)
            # label = data['label'].reshape(prob.shape[0]*prob.shape[1])
            # print(p.shape)
            # print(label.shape)
            loss_d = self.criterion_d(prob, data['label'].float())
            loss_c = self.criterion_c(out.transpose(1, 2), data["origin_text"])
            loss = self.gama * loss_c + (1-self.gama) * loss_d

            if train:
                self.optim_schedule.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_schedule.step_and_update_lr()

            correct = out.argmax(dim=-1).eq(data["origin_text"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["label"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)
        return avg_loss / len(data_iter)


class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len=512, pad_first=True):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len
        self.data_size = len(dataset)
        self.pad_first = pad_first

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        origin_text = item['origin_text']
        random_text = item['random_text']
        label = item['label']

        origin_text = [self.tokenizer.convert_tokens_to_ids([x])[0] for x in origin_text]
        random_text = [self.tokenizer.convert_tokens_to_ids([x])[0] for x in random_text]
        label = [int(x) for x in label.split(' ')]

        pad_len = self.max_len - len(origin_text) - 2 if self.max_len-2 > len(origin_text) else 0
        if pad_len == 0:
            origin_text = origin_text[:self.max_len-2]
            random_text = random_text[:self.max_len - 2]
            label = label[:self.max_len - 2]

        if self.pad_first:
            origin_text = [self.tokenizer.cls_token_id]\
                          + [self.tokenizer.pad_token_id] * pad_len + origin_text\
                          + [self.tokenizer.sep_token_id]

            random_text = [self.tokenizer.cls_token_id]\
                          + [self.tokenizer.pad_token_id] * pad_len + random_text\
                          + [self.tokenizer.sep_token_id]

            label = [self.tokenizer.pad_token_id]\
                    + [self.tokenizer.pad_token_id] * pad_len + label\
                    + [self.tokenizer.pad_token_id]

        else:
            origin_text = [self.tokenizer.cls_token_id]\
                          + origin_text + [self.tokenizer.pad_token_id] * pad_len\
                          + [self.tokenizer.sep_token_id]

            random_text = [self.tokenizer.cls_token_id]\
                          + random_text + [self.tokenizer.pad_token_id] * pad_len\
                          + [self.tokenizer.sep_token_id]

            label = [self.tokenizer.pad_token_id]\
                    + label + [self.tokenizer.pad_token_id] * pad_len\
                    + [self.tokenizer.pad_token_id]
        output = {
            'origin_text': origin_text,
            'random_text': random_text,
            'label': label
        }

        return {key: torch.tensor(value) for key, value in output.items()}

if __name__ == '__main__':
    dataset = pd.read_csv('data/processed_data/all_same_765376/train.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=5, shuffle=True)
    for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
        print('Start train {} ford'.format(k))
        config = BertConfig.from_pretrained('data/chinese_wwm_pytorch/bert_config.json')
        tokenizer = BertTokenizer.from_pretrained('data/chinese_wwm_pytorch/vocab.txt')
        bert = BertModel.from_pretrained('data/chinese_wwm_pytorch/pytorch_model.bin', config=config)

        train = dataset.iloc[train_index]
        val = dataset.iloc[val_index]
        train_dataset = BertDataset(tokenizer, train, max_len=10)
        train_data_loader = DataLoader(train_dataset, batch_size=320, num_workers=2)
        val_dataset = BertDataset(tokenizer, val, max_len=512)
        val_data_loader = DataLoader(val_dataset, batch_size=320, num_workers=2)
        trainer = SoftMaskedBertTrainer(bert, tokenizer, device)
        best_loss = 100000
        for e in range(10):
            trainer.train(train_data_loader, e)
            val_loss = trainer.evaluate(val_data_loader, e)
            if best_loss > val_loss:
                best_loss = val_loss
                trainer.save('best_model_{}ford.pt'.format(k))
                print('Best val loss {}'.format(best_loss))

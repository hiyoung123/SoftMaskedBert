#!usr/bin/env python
#-*- coding:utf-8 -*-

import os
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SoftMaskedBertTrainer():
    def __init__(self, bert, tokenizer, device, hidden=256, layer_n=1, lr=2e-5, gama=0.8, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=10000):

        self.device = device
        self.tokenizer = tokenizer
        self.model = SoftMaskedBert(bert, self.tokenizer, hidden, layer_n, self.device).to(self.device)

        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for train" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[0,1,2])

        optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(optim, hidden, n_warmup_steps=warmup_steps)
        self.criterion_c = nn.NLLLoss()
        self.criterion_d = nn.BCELoss()
        self.gama = gama
        self.log_freq = 100

    def train(self, train_data, epoch):
        self.model.train()
        return self.iteration(epoch, train_data)

    def evaluate(self, val_data, epoch):
        self.model.eval()
        return self.iteration(epoch, val_data, train=False)

    def inference(self, data_loader):
        self.model.eval()
        out_put = []
        data_loader = tqdm.tqdm(enumerate(data_loader),
                              desc="%s" % 'Inference:',
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(data["input_ids"], data["input_mask"], data["segment_ids"]) #prob [batch_size, seq_len, 1]
            out_put.extend(out.argmax(dim=-1).cpu().numpy().tolist())
        return [''.join(self.tokenizer.convert_ids_to_tokens(x)) for x in out_put]

    def save(self, file_path):
        torch.save(self.model.cpu(), file_path)
        self.model.to(self.device)
        print('Model save {}'.format(file_path))

    def load(self, file_path):
        if not os.path.exists(file_path):
            return
        self.model = torch.load(file_path)
        self.model.to(self.device)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "val"

        # Setting the tqdm progress bar
        data_loader = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        # total_correct = 0
        total_element = 0
        c_correct = 0
        d_correct = 0

        for i, data in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            out, prob = self.model(data["input_ids"], data["input_mask"], data["segment_ids"]) #prob [batch_size, seq_len, 1]
            prob = prob.reshape(-1, prob.shape[1])
            loss_d = self.criterion_d(prob, data['label'].float())
            loss_c = self.criterion_c(out.transpose(1, 2), data["output_ids"])
            loss = self.gama * loss_c + (1-self.gama) * loss_d

            if train:
                self.optim_schedule.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_schedule.step_and_update_lr()

            # correct = out.argmax(dim=-1).eq(data["output_ids"]).sum().item()
            out = out.argmax(dim=-1)
            c_correct += sum([out[i].equal(data['output_ids'][i]) for i in range(len(out))])
            prob = torch.round(prob).long()
            d_correct += sum([prob[i].equal(data['label'][i]) for i in range(len(prob))])

            avg_loss += loss.item()
        #     total_correct += c_correct
        #     # total_element += data["label"].nelement()
            total_element += len(data)

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "d_acc": d_correct / total_element,
                "c_acc": c_correct / total_element
            }

            if i % self.log_freq == 0:
                data_loader.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_loader), "d_acc=",
              d_correct / total_element, "c_acc", c_correct / total_element)
        return avg_loss / len(data_loader)


class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len=512, pad_first=True, mode='train'):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len
        self.data_size = len(dataset)
        self.pad_first = pad_first
        self.mode = mode

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        input_ids = item['random_text']
        input_ids = ['[CLS]'] + list(input_ids)[:min(len(input_ids), self.max_len - 2)] + ['[SEP]']
        # convert to bert ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if self.pad_first:
            input_ids = [0] * pad_len + input_ids
            input_mask = [0] * pad_len + input_mask
            segment_ids = [0] * pad_len + segment_ids
        else:
            input_ids = input_ids + [0] * pad_len
            input_mask = input_mask + [0] * pad_len
            segment_ids = segment_ids + [0] * pad_len

        output = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        if self.mode == 'train':
            output_ids = item['origin_text']
            label = item['label']
            label = [int(x) for x in label if x != ' ']
            output_ids = ['[CLS]'] + list(output_ids)[:min(len(output_ids), self.max_len - 2)] + ['[SEP]']
            label = [0] + label[:min(len(label), self.max_len - 2)] + [0]

            output_ids = self.tokenizer.convert_tokens_to_ids(output_ids)
            pad_label_len = self.max_len - len(label)
            if self.pad_first:
                output_ids = [0] * pad_len + output_ids
                label = [0] * pad_label_len + label
            else:
                output_ids = output_ids + [0] * pad_len
                label = label + [0] * pad_label_len

            output = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'output_ids': output_ids,
                'label': label
            }
        return {key: torch.tensor(value) for key, value in output.items()}

if __name__ == '__main__':
    dataset = pd.read_csv('data/processed_data/all_same_765376/train.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained('data/chinese_wwm_pytorch/bert_config.json')
    tokenizer = BertTokenizer.from_pretrained('data/chinese_wwm_pytorch/vocab.txt')
    kf = KFold(n_splits=5, shuffle=True)
    for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
        print('Start train {} ford'.format(k))
        bert = BertModel.from_pretrained('data/chinese_wwm_pytorch/pytorch_model.bin', config=config)

        train = dataset.iloc[train_index]
        val = dataset.iloc[val_index]
        train = BertDataset(tokenizer, train, max_len=152)
        train = DataLoader(train, batch_size=8, num_workers=2)
        val = BertDataset(tokenizer, val, max_len=152)
        val = DataLoader(val, batch_size=8, num_workers=2)
        trainer = SoftMaskedBertTrainer(bert, tokenizer, device)
        best_loss = 100000
        for e in range(100):
            trainer.train(train, e)
            val_loss = trainer.evaluate(val, e)
            if best_loss > val_loss:
                best_loss = val_loss
                trainer.save('best_model_{}ford.pt'.format(k))
                print('Best val loss {}'.format(best_loss))

            trainer.load('best_model_{}ford.pt'.format(k))
            for i in trainer.inference(val):
                print(i)
                print('\n')
import os.path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from config import get_config
from data import load_dataset
from model import BM_FNN


class CPT:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('【main.py】> creating model {}'.format(args.model_name))
        # Create model
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/bert')
            self.base_model = AutoModel.from_pretrained('./model/bert')
            self.base_config = AutoConfig.from_pretrained('./model/bert')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/roberta')
            self.base_model = AutoModel.from_pretrained('./model/roberta')
            self.base_config = AutoConfig.from_pretrained('./model/roberta')
        elif args.model_name == 'codebert':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/codebert')
            self.base_model = AutoModel.from_pretrained('./model/codebert')
            self.base_config = AutoConfig.from_pretrained('./model/codebert')
        elif args.model_name == "codet5":
            self.tokenizer = AutoTokenizer.from_pretrained('./model/codet5')
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained('./model/codet5')
            self.base_config = AutoConfig.from_pretrained('./model/codet5')
        elif args.model_name == 'albert':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/albert')
            self.base_model = AutoModel.from_pretrained('./model/albert')
            self.base_config = AutoConfig.from_pretrained('./model/albert')
        elif args.model_name == 'distilbert':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/distilbert')
            self.base_model = AutoModel.from_pretrained('./model/distilbert')
            self.base_config = AutoConfig.from_pretrained('./model/distilbert')
        elif args.model_name == 'graphcodebert':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/graphcodebert')
            self.base_model = AutoModel.from_pretrained('./model/graphcodebert')
            self.base_config = AutoConfig.from_pretrained('./model/graphcodebert')
        elif args.model_name == 't5':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/t5')
            self.base_model = AutoModel.from_pretrained('./model/t5')
            self.base_config = AutoConfig.from_pretrained('./model/t5')
        elif args.model_name == 'codet5+':
            self.tokenizer = AutoTokenizer.from_pretrained('./model/codet5+', trust_remote_code=True)
            self.base_model = AutoModel.from_pretrained('./model/codet5+', trust_remote_code=True)
            self.base_config = AutoConfig.from_pretrained('./model/codet5+', trust_remote_code=True)
        else:
            raise ValueError('unknown model')

        self.Mymodel = BM_FNN(args, self.base_model, self.tokenizer, self.base_config, args.num_classes)
        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('【main.py】 > cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('【main.py】 > training arguments:')
        for arg in vars(self.args):
            self.logger.info(f"【main.py】 > {arg}: {getattr(self.args, arg)}")

    def get_score(self, pre, lab):
        TP, TN, FP, FN = 0, 0, 0, 0
        for i, j in zip(pre, lab):
            if i == 1 and j == 1:
                TP += 1
            elif i == 1 and j == 0:
                FP += 1
            elif i == 0 and j == 1:
                FN += 1
            elif i == 0 and j == 0:
                TN += 1
        acc = (TP + TN) * 100 / (TP + TN + FP + FN)
        try:
            precision = TP * 100 / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = TP * 100 / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        return TP, TN, FP, FN, acc, precision, recall, f1

    def _train(self, args, dataloader, criterion, optimizer, epoch):
        total_loss = 0
        predicts = []
        labels = []
        self.Mymodel.train()
        for batch in tqdm(dataloader, ascii='>='):
            in_, lab, loc, id = batch
            lab = lab.to(args.device)
            pre = self.Mymodel(in_)
            loss = criterion(pre, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * lab.size(0)
            predicts.extend(torch.argmax(pre, dim=1))
            labels.extend(lab)
        torch.save(self.Mymodel, os.path.join(os.path.join('model_save', args.save_path), str(epoch) + "_model.pth"))
        return total_loss / len(dataloader), self.get_score(predicts, labels)

    def _valid(self, args, dataloader, criterion, optimizer):
        total_loss = 0
        predicts = []
        probs = []
        labels = []
        self.Mymodel.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, ascii='>='):
                in_, lab, loc, id = batch
                lab = lab.to(args.device)
                pre = self.Mymodel(in_)
                loss = criterion(pre, lab)
                total_loss += loss.item() * lab.size(0)
                probs.extend(nn.functional.softmax(pre, dim=1))
                # print(probs)
                predicts.extend(torch.argmax(pre, dim=1))
                labels.extend(lab)
            return total_loss / len(dataloader), self.get_score(predicts, labels)

    def run(self):
        train_dataloader = load_dataset(args=args, tokenizer=self.tokenizer, logger=self.logger, path=args.train_path)
        valid_dataloader = load_dataset(args=args, tokenizer=self.tokenizer, logger=self.logger, path=args.test_path)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        for epoch in range(self.args.num_epoch):
            train_loss, (TP, TN, FP, FN, train_acc, train_precision, train_recall, train_f1) = self._train(args, train_dataloader, criterion, optimizer, epoch)
            self.logger.info('【main.py】[train]--Epoch: {:d}, loss: {:.4f}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}, TP: {:d}, TN: {:d}, FP: {:d}, FN: {:d}'.format(
                epoch, train_loss, train_acc, train_precision, train_recall, train_f1, TP, TN, FP, FN))
            valid_loss, (TP, TN, FP, FN, valid_acc, valid_precision, valid_recall, valid_f1) = self._valid(args, valid_dataloader, criterion, optimizer)
            self.logger.info('【main.py】[valid]--Epoch: {:d}, loss: {:.4f}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}, TP: {:d}, TN: {:d}, FP: {:d}, FN: {:d}'.format(
                epoch, valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, TP, TN, FP, FN))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    cpt = CPT(args, logger)
    cpt.run()

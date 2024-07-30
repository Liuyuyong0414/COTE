import json
import math
import os.path
import time
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from config import get_config
from data import load_dataset
from model import BM_FNN
import numpy as np


class CPT:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('【main.py】> creating model {}'.format(args.model_name))
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

    def get_score(self, pre, lab, ids):
        TP, TN, FP, FN = 0, 0, 0, 0
        FP_id = []
        FN_id = []
        for i, j, id in zip(pre, lab, ids):
            if i == 1 and j == 1:
                TP += 1
            elif i == 1 and j == 0:
                FP += 1
                FP_id.append(int(id))
            elif i == 0 and j == 1:
                FN += 1
                FN_id.append(int(id))
            elif i == 0 and j == 0:
                TN += 1
        with open("./data/FP_id.json", "w") as f:
            json.dump(FP_id, f, indent=4)
        with open("./data/FN_id.json", "w") as f:
            json.dump(FN_id, f, indent=4)
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

    def _valid(self, args, dataloader):
        total_loss = 0
        probs = []
        labels = []
        locs = []
        ids = []
        self.Mymodel.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, ascii='>='):
                in_, lab, loc, id = batch
                lab = lab.to(args.device)
                try:
                    pre = self.Mymodel(in_)
                except:
                    continue
                locs.extend(loc.cpu().numpy())
                ids.extend(id.cpu().numpy())
                probs.extend(nn.functional.softmax(pre, dim=1)[:, 1].cpu().numpy())
                labels.extend(lab.cpu().numpy())

            predicts = list((np.array(probs) >= 0.5).astype(int))
            probs = [float(x) for x in probs]
            labels = [int(x) for x in labels]
            with open("probs.json", "w") as f:
                json.dump(probs, f, indent=4)
            with open("labels.json", "w") as f:
                json.dump(labels, f, indent=4)
            roc_auc = roc_auc_score(labels, probs) * 100
            ap = average_precision_score(labels, probs) * 100
            return total_loss / len(dataloader), self.get_score(predicts, labels, ids), roc_auc, ap

    def run(self):
        epoch = 3
        model_path = "./model_save/A-codet5_fnn_24-07-16_00-23-59/" + str(epoch) + "_model.pth"
        self.Mymodel = torch.load(model_path)
        for path in ["../data/test.json"]:
            dataloader = load_dataset(args=args, tokenizer=self.tokenizer, logger=self.logger, path=path)
            start_time = time.time()
            loss, (TP, TN, FP, FN, acc, precision, recall, f1), auc, ap = self._valid(args, dataloader)
            self.logger.info(
                '【main.py】[valid]--Epoch: {:d}, loss: {:.4f}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}, Auc: {:.2f}, ap: {:.2f}, TP: {:d}, TN: {:d}, FP: {:d}, FN: {:d}'.format(
                    epoch, loss, acc, precision, recall, f1, auc, ap, TP, TN, FP, FN))
            print("Prediction time： ", time.time() - start_time)


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    cpt = CPT(args, logger)
    cpt.run()

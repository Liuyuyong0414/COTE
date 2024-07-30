import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta

import torch


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='bert',
                        choices=['bert', 'codebert', 'roberta', 'codet5', 'codet5+', 'albert', 'distilbert', 't5', 'bart', 'graphcodebert'])
    parser.add_argument('--method_name', type=str, default='fnn')
    parser.add_argument('--train_path', type=str, default="../data/train.json")
    parser.add_argument('--test_path', type=str, default="../data/test.json")
    parser.add_argument('--parser_jMethod_jar', type=str, default="./data/parser_method.jar")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))

    args = parser.parse_args()
    args.device = torch.device(args.device)

    args.save_path = '{}_{}_{}'.format(args.model_name, args.method_name, (datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d_%H-%M-%S')[2:])
    args.log_name = args.save_path + ".log"
    if not os.path.exists(os.path.join('model_save', args.save_path)):
        os.makedirs(os.path.join('model_save', args.save_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join(os.path.join('model_save', args.save_path), args.log_name)))
    return args, logger

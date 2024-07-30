from functools import partial
import time
import ijson
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from code_to_seq import operate_seq


class MyDataset(Dataset):
    def __init__(self, cc, pcc, tuc, label, loc, id):
        self.loc = loc
        self.label = label
        self.pcc = pcc
        self.tuc = tuc
        self.cc = cc
        self.id = id

    def __getitem__(self, index):
        return (self.cc[index], self.pcc[index], self.tuc[index], self.label[index], self.loc[index], self.id[index])

    def __len__(self):
        return len(self.loc)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = next(ijson.items(file, ""), None)
    return data


def analysis(logger, args, datas):
    candidate = 0
    pa = 0
    production_code_change_lst = []
    test_unit_code_lst = []
    commit_cmg_lst = []
    loc_lst = []
    label_lst = []
    id_lst = []
    for data, label in tqdm(datas):
        if label != 0:
            label = 1
        commit_cmg = data["commit_message"]
        edit_seq = operate_seq(data["lpfc"], data["rpfc"])
        production_code_change = " ".join(item for sublist in edit_seq for item in sublist)
        test_unit_code = data["ltfc"]
        commit_cmg_lst.append(commit_cmg)
        production_code_change_lst.append(production_code_change)
        test_unit_code_lst.append(test_unit_code)
        loc_lst.append(len(data["ltfc"].splitlines()))
        label_lst.append(label)
        id_lst.append(data["id"])
    logger.info(f"【data.py】No Candidate: {candidate}, Pa number: {pa}, Last number: {len(commit_cmg_lst)}")
    return commit_cmg_lst, production_code_change_lst, test_unit_code_lst, label_lst, loc_lst, id_lst


def get_token(batch, args, tokenizer):
    encoded_inputs = []
    labels = []
    locs = []
    ids = []
    if args.model_name in ['codet5', 'bart', 'codet5+', 't5']:
        for i, j, k, l, m, n in batch:
            # Assuming i, j, k are strings and l, m are labels and masks
            encoded_i = tokenizer.encode(i, max_length=args.max_length, padding='max_length', truncation=True)
            encoded_j = tokenizer.encode(j, max_length=args.max_length, padding='max_length', truncation=True)
            encoded_k = tokenizer.encode(k, max_length=args.max_length, padding='max_length', truncation=True)

            encoded_inputs.append(encoded_i + encoded_j + encoded_k)
            labels.append(l)
            locs.append(m)
            ids.append(n)

        return torch.tensor(encoded_inputs), torch.tensor(labels), torch.tensor(locs), torch.tensor(ids)
    else:
        i, j, k, l, m, n = map(list, zip(*batch))
        encoded_i = tokenizer(i, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoded_j = tokenizer(j, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoded_k = tokenizer(k, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return (encoded_i, encoded_j, encoded_k), torch.tensor(l), torch.tensor(m), torch.tensor(n)


def load_dataset(args, tokenizer, logger, path):
    logger.info("【data.py】 Loading " + path)
    data = read_json(path)
    start_time = time.time()
    cc, pcc, tuc, label, loc, id = analysis(logger, args, data)
    dataset = MyDataset(cc, pcc, tuc, label, loc, id)
    collate_fn = partial(get_token, args=args, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    logger.info("【data.py】 Loading Success!")
    print("data_time： ", time.time() - start_time)
    return dataloader

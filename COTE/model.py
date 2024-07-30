import math

import torch
import torch.nn.functional as F
from torch import nn


# Bert + FNN

class BM_FNN(nn.Module):
    def __init__(self, args, model, tokenizer, config, num_classes):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.num_classes = num_classes
        if self.args.model_name in ['codet5', 'bart', 'codet5+', 't5']:
            self.hidden_size = config.hidden_size
        else:
            self.hidden_size = 3 * config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        for param in self.model.parameters():
            param.requires_grad = (True)

    def get_output(self, inputs):
        if self.args.model_name in ['codet5', 'bart']:
            inputs = inputs.to(self.args.device)
            attention_mask = inputs.ne(self.tokenizer.pad_token_id)
            outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=inputs, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = inputs.eq(self.config.eos_token_id)
            out = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
            return out
        elif self.args.model_name in ['t5', 'codet5+']:
            inputs = inputs.to(self.args.device)
            attention_mask = inputs.ne(self.tokenizer.pad_token_id)
            outputs = self.model(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=inputs, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = inputs.eq(self.config.eos_token_id)
            out = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
            return out
        else:
            i, j, k = inputs
            i = {K: V.to(self.args.device) for K, V in i.items()}
            j = {K: V.to(self.args.device) for K, V in j.items()}
            k = {K: V.to(self.args.device) for K, V in k.items()}
            i_out, j_out, k_out = self.model(i["input_ids"], i["attention_mask"]), self.model(j["input_ids"], j["attention_mask"]), self.model(k["input_ids"], k["attention_mask"])
            out = torch.cat((i_out[0][:, -1, :], j_out[0][:, -1, :], k_out[0][:, -1, :]), dim=1)
            return out

    def forward(self, in_):
        in_ = self.get_output(in_)
        out = self.fc(in_)
        return out

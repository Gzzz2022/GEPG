import json
import random

import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''

    def __init__(self, data_name: str, knowledge_num: int, batch_size: int = 256, test: bool = False):
        self.batch_size = batch_size
        self.ptr = 0
        self.data = []
        self.is_test = test

        if test:
            data_file = f'./data/{data_name}/train_set.json'
        else:
            data_file = f'./data/{data_name}/test_set.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

        self.knowledge_dim = knowledge_num

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
            y = log['score']
            input_stu_ids.append(log['stu_id'])
            input_exer_ids.append(log['exer_id'])
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowedge_embs), torch.LongTensor(ys)

    def get_batch_num(self):
        return len(self.data) // self.batch_size

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
        
        if not self.is_test:
            random.shuffle(self.data)


class ValTestDataLoader(object):
    def __init__(self, data_name: str, knowledge_num: int):
        self.ptr = 0
        self.data = []

        data_file = f'./data/{data_name}/test_set.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)

        self.knowledge_dim = knowledge_num

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        stu_id = self.data[self.ptr]['stu_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(stu_id)
            input_exer_ids.append(log['exer_id'])
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

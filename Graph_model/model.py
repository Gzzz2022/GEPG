import torch
import torch.nn as nn
from .fusion import Fusion


class Net(nn.Module):
    def __init__(self, args, local_map):
        super(Net, self).__init__()

        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

        self.knowledge_num = args.knowledge_n
        self.exer_num = args.exer_n
        self.stu_num = args.student_n

        self.emb = args.emb

        self.local_map = {

            'undirected_g': local_map['undirected_g'].to(self.device),
            'e_to_k': local_map['e_to_k'].to(self.device),
            'k_to_e': local_map['k_to_e'].to(self.device),
            'e_to_u': local_map['e_to_u'].to(self.device),
            'u_to_e': local_map['u_to_e'].to(self.device)
        }

        self.knowledge_emb = nn.Embedding(self.knowledge_num, self.emb)
        self.exercise_emb = nn.Embedding(self.exer_num, self.emb)
        self.student_emb = nn.Embedding(self.stu_num, self.emb)

        self.k_index = torch.LongTensor(list(range(self.knowledge_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.stu_num))).to(self.device)

        self.FusionLayer1 = Fusion(args, self.local_map)
        self.FusionLayer2 = Fusion(args, self.local_map)

        self.prednet_full1 = nn.Linear(2 * self.emb, self.emb, bias=False)

        self.prednet_full2 = nn.Linear(2 * self.emb, self.emb, bias=False)

        self.prednet_full3 = nn.Linear(1 * self.emb, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb, exer_emb, all_stu_emb = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)

        kn_emb, exer_emb, all_stu_emb = self.FusionLayer2(kn_emb, exer_emb, all_stu_emb)

        batch_stu_emb = all_stu_emb[stu_id]
        batch_stu_vector = batch_stu_emb.unsqueeze(1).expand(-1, self.knowledge_num, -1)

        batch_exer_emb = exer_emb[exer_id]
        batch_exer_vector = batch_exer_emb.unsqueeze(1).expand(-1, self.knowledge_num, -1)

        kn_vector = kn_emb.unsqueeze(0).expand(batch_stu_emb.shape[0], -1, -1)

        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

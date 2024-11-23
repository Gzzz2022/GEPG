import torch
import torch.nn as nn
import torch.nn.functional as F
from .GraphLayer import GraphLayer


class Fusion(nn.Module):
    def __init__(self, args, local_map):
        super(Fusion, self).__init__()

        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.exer_n = args.exer_n
        self.emb = args.emb

        self.undirected_g = local_map['undirected_g']
        self.e_to_k = local_map['e_to_k']
        self.k_to_e = local_map['k_to_e']
        self.e_to_u = local_map['e_to_u']
        self.u_to_e = local_map['u_to_e']

        self.undirected_gat = GraphLayer(self.undirected_g, self.emb, self.emb)

        self.e_to_k = GraphLayer(self.e_to_k, self.emb, self.emb)
        self.k_to_e = GraphLayer(self.k_to_e, self.emb, self.emb)

        self.e_to_u = GraphLayer(self.e_to_u, self.emb, self.emb)
        self.u_to_e = GraphLayer(self.u_to_e, self.emb, self.emb)

        self.k_attn_fc1 = nn.Linear(2 * self.emb, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * self.emb, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * self.emb, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * self.emb, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * self.emb, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        e_to_k_graph = self.e_to_k(e_k_graph)
        k_to_e_graph = self.k_to_e(e_k_graph)

        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        e_to_u_graph = self.e_to_u(e_u_graph)
        u_to_e_graph = self.u_to_e(e_u_graph)

        A = kn_emb

        C = k_undirected
        D = e_to_k_graph[self.exer_n:]

        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)

        score2 = self.k_attn_fc2(concat_c_2)
        score3 = self.k_attn_fc3(concat_c_3)
        score = F.softmax(torch.cat([score2, score3], dim=1), dim=1)

        kn_emb = A + score[:, 0].unsqueeze(1) * C + score[:, 1].unsqueeze(1) * D

        A = exer_emb
        B = k_to_e_graph[0: self.exer_n]
        C = u_to_e_graph[0: self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        all_stu_emb = all_stu_emb + e_to_u_graph[self.exer_n:]

        return kn_emb, exer_emb, all_stu_emb

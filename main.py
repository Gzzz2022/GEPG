import os
import random
import time
from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import torch
import tqdm
from torch.nn import Embedding

import pickle
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from Graph_model.model import Net
from ctl import CommonArgParser
from build_model import construct_local_map
from Graph_model.data_loader import TrainDataLoader
from EPG.Rainbow import Rainbow
from EPG.Qlearning import DQNAgent
from EPG.prioritized_replay import PrioritizedReplay
from build_model import adjust_args

import csv

import warnings

warnings.filterwarnings("ignore")

args = CommonArgParser().parse_args()
adjust_args(args)
device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')


def get_state(paper):
    paper = torch.sort(paper).values
    paper_state = torch.tensor([], dtype=torch.float32).to(paper.device)
    interaction_emb = Embedding(args.knowledge_n * 2, args.q_embedding_size)
    for p in paper:
        paper_state = torch.cat((paper_state, torch.mean(interaction_emb(torch.tensor(data_dict[p.item()])), dim=0)))
    return paper_state


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def add_a_in_paper_dict(new_key, new_value, index, my_dict):
    keys = list(my_dict.keys())
    values = list(my_dict.values())

    keys.insert(index, new_key)
    values.insert(index, new_value)
    my_dict = dict(zip(keys, values))
    return my_dict


def replace_a_in_paper_dict(new_key, new_value, index, my_dict):
    keys = list(my_dict.keys())
    values = list(my_dict.values())
    keys[index] = new_key
    values[index] = new_value
    new_dict = dict(zip(keys, values))
    return new_dict


def get_student(stu_num):
    np.random.seed(42)
    students = np.arange(args.student_n)
    u_list = np.random.choice(students, stu_num, replace=False)
    return u_list


def load_snapshot(model, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def get3data(u_list, pa_list, dict):
    input_stu_ids, input_exer_ids, input_knowedge_embs = u_list, pa_list, []

    for key in input_exer_ids:
        knowledge_emb = [0.] * args.knowledge_n
        values = dict[key]
        for v in values:
            knowledge_emb[v] = 1.0
        input_knowedge_embs.append(knowledge_emb)

    return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs)


def get_all_dir():
    data_dict = {}
    data_file = f'./data/{args.data_name}/graph/e_to_k.txt'
    with open(data_file, 'r') as file:
        for line in file:
            line = line.strip()
            e, k = line.split('\t')
            e = int(e)
            k = int(k) - args.exer_n
            if e in data_dict:
                data_dict[e].append(k)
            else:
                data_dict[e] = [k]
    return data_dict


def get_paper_dict(pa_list, dict):
    p_dict = {}
    for e in pa_list:
        if e in dict:
            p_dict[e] = dict[e]

    return p_dict


def get_cover(input_dict, index=None):
    knowledge_counts = [0] * args.knowledge_n

    for key, value in input_dict.items():
        if index is not None and key == index:
            continue
        for kwlg in value:
            knowledge_counts[kwlg] += 1

    sum_k = sum(knowledge_counts)
    cover = [count / sum_k for count in knowledge_counts]
    return cover


def get_stu4index_scores(input_stu_ids, input_exer_ids, input_knowledge_embs):
    with torch.no_grad():
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)

    return output


def get_stu4ex_scores(input_stu_ids, input_exer_ids, input_knowledge_embs):
    score_list = []
    stu4ex_scores = []
    num_st = len(input_stu_ids)
    num_ex = len(input_exer_ids)

    for i in tqdm.tqdm(range(num_st)):
        stu4ex = []
        input_stu_id = input_stu_ids[i].expand(num_ex)

        with torch.no_grad():
            output = net.forward(input_stu_id, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)

            output_list = output.tolist()
            stu4ex_scores.append(output_list)

            mark = torch.sum(output)
            score_list.append(mark.item())

    return stu4ex_scores, score_list


def get_reward(stu4ex_scores, scores_list, paper_dict, index=None):
    len_paper = len(paper_dict)
    len_scores_list = len(scores_list)

    r1 = 1 - abs(sum(scores_list) / len_scores_list - 70) / len_paper

    X = stats.truncnorm((0 - 70) / 15, (100 - 70) / 15, loc=70, scale=15)
    paper_distribution = X.rvs(100, random_state=seed)
    r2 = 1 - stats.wasserstein_distance(paper_distribution, scores_list) / len_paper

    scores_list.sort()
    l1 = int(0.27 * len_scores_list)
    s1_dh = scores_list[:l1]
    s1_dl = scores_list[-l1:]
    a1 = sum(s1_dh) / len(s1_dh)
    a2 = sum(s1_dl) / len(s1_dl)
    r3 = (a2 - a1) / 100

    paper_cover = get_cover(paper_dict, index)
    data_cover = get_cover(data_dict)
    r4 = cosine_similarity([data_cover], [paper_cover])[0][0]

    r = r1 / 4 + r2 / 4 + r3 / 4 + r4 / 4
    return r, r1, r2, r3, r4


def ExamPaperGeneration(data_dict):
    for paperindex in range(1, 101):
        print(f"paper {paperindex}")

        done = False

        paper = torch.tensor(random.sample(QB, 100), dtype=torch.long)
        paper_list = paper.tolist()
        paper_dict = get_paper_dict(paper_list, data_dict)
        input_stu_ids, input_exer_ids, input_knowledge_embs = get3data(select_u_list, paper_list, paper_dict)
        input_stu_ids, input_exer_ids, input_knowledge_embs = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device)

        stu4ex_scores, score_list = get_stu4ex_scores(input_stu_ids, input_exer_ids, input_knowledge_embs)

        r_random_12345 = get_reward(stu4ex_scores, score_list, paper_dict)
        r_ = r_random_12345[0]

        paper_state = get_state(paper).to(device)
        stu4ex_scores = torch.tensor(stu4ex_scores, dtype=torch.float32)

        paper = paper.detach().cpu().numpy().tolist()
        best_paper = paper
        max_reward = r_
        best_paper_r = r_random_12345
        score_list_best_paper = score_list
        stu4ex_scores_best_paper = stu4ex_scores
        best_paper_dict = paper_dict

        for episode in tqdm.tqdm(range(args.max_episode), desc=f'episode: {paperindex}'):
            y_suit = []

            for index in range(100):
                mask = torch.ones(stu4ex_scores.shape)
                mask[:, index] = 0
                stu_scores = torch.sum(stu4ex_scores * mask, dim=-1)
                r_shanyiti = get_reward(None, stu_scores.detach().cpu().numpy(),
                                        paper_dict, paper[index])[0]
                y_suit.append(r_shanyiti)

            replace_index = np.argmax(y_suit)

            a = agent.select_action(paper_state, QB)

            while a in paper:
                a = agent.select_action(paper_state, QB)

            paper_ = paper.copy()
            paper_[replace_index] = a

            paper__dict = replace_a_in_paper_dict(a, data_dict[a], replace_index, paper_dict)

            input_stu_ids = torch.LongTensor(select_u_list).to(device)
            input_exer_ids = torch.LongTensor([a] * stu_num).to(device)
            knowledge_emb = [0.] * args.knowledge_n
            values = data_dict[a]
            for v in values:
                knowledge_emb[v] = 1.0

            input_knowledge_embs = torch.tensor(knowledge_emb, dtype=int).unsqueeze(0).expand(stu_num, -1).to(device)

            stu2a_scores = get_stu4index_scores(input_stu_ids, input_exer_ids, input_knowledge_embs)

            stu4ex_scores_paper_ = stu4ex_scores.clone()
            stu4ex_scores_paper_[:, replace_index] = stu2a_scores
            score_list_paper_ = torch.sum(stu4ex_scores_paper_, dim=-1)

            r_after_12345 = get_reward(stu4ex_scores_paper_, score_list_paper_.detach().cpu().numpy(), paper__dict)
            r_ = r_after_12345[0]

            if r_ > max_reward:
                best_paper = paper_
                max_reward = r_
                best_paper_r = r_after_12345
                score_list_best_paper = score_list_paper_
                stu4ex_scores_best_paper = stu4ex_scores_paper_
                best_paper_dict = paper__dict

            paper_state_ = get_state(torch.tensor(paper_))

            if (episode + 1) % 200 == 0:
                print(f"episode: {episode}, reward: {r_}")

            agent.store_transition(paper_state, a, r_, paper_state_, done)

            if agent.memory_counter > agent.batch_size:
                agent.learn()

            paper = paper_
            stu4ex_scores = stu4ex_scores_paper_
            paper_dict = paper__dict
            paper_state = paper_state_
            done = True

        print(f"The optimal test paper at the current episode is {best_paper}, The maximum reward is {best_paper_r}")

        path = f'./papers/{args.data_name}'
        with open(os.path.join(path, f"best_paper_{paperindex}.pkl"), "wb") as f:
            pickle.dump(best_paper, f)

        csv_file = os.path.join(path, 'GEPG.csv')

        headers = ['r', 'r1', 'r2', 'r3', 'r4']

        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(headers)

            writer.writerow(best_paper_r)


def load_graph(filename):
    graph = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            node1, node2 = map(int, line.split())

            if node1 in graph:
                graph[node1].append(node2)
            else:
                graph[node1] = [node2]
            if node2 in graph:
                graph[node2].append(node1)
            else:
                graph[node2] = [node1]
    return graph


def choose_by_node():
    if not os.path.exists(f'./papers/{args.data_name}/best_paper_1.pkl'):
        raise FileNotFoundError(f'./papers/{args.data_name}/best_paper_1.pkl')

    with open(f'./papers/{args.data_name}/best_paper_1.pkl', "rb") as f:
        old_paper = pickle.load(f)

    old_paper_list = old_paper
    old_paper_dict = get_paper_dict(old_paper_list, data_dict)

    k2e_dict = {}
    with open(f'./data/{args.data_name}/graph/k_to_e.txt', 'r') as file:
        for line in file:
            line = line.strip()
            k, e = line.split('\t')

            k = int(k) - args.exer_n
            e = int(e)
            if k in k2e_dict:
                k2e_dict[k].append(e)
            else:
                k2e_dict[k] = [e]

    parallel_paper_list = deepcopy(old_paper_list)
    parallel_paper_dict = deepcopy(old_paper_dict)

    graph = load_graph(f"./data/{args.data_name}/graph/K_Undirected.txt")

    n = len(old_paper_list)

    for i in range(n):
        kn = data_dict[parallel_paper_list[i]]

        if len(kn) == 1:

            ex = k2e_dict[kn[0]]
            ex.remove(parallel_paper_list[i])
            if len(ex) > 0:

                selected_p = random.choice(ex)
                while True:
                    if selected_p not in parallel_paper_list:
                        break
                    else:
                        selected_p = random.choice(ex)

                selected_kn = data_dict[selected_p]

                parallel_paper_list[i] = selected_p
                parallel_paper_dict = replace_a_in_paper_dict(selected_p, selected_kn, i, parallel_paper_dict)
            else:

                neibor_kns = graph[kn[0]]

                if len(neibor_kns) == 0:

                    continue
                else:

                    random_KN = random.choice(neibor_kns)
                    random_EX = random.choice(k2e_dict[random_KN])
                    while True:
                        if random_EX not in parallel_paper_list:
                            break
                        else:

                            random_KN = random.choice(neibor_kns)
                            random_EX = random.choice(k2e_dict[random_KN])

                    parallel_paper_list[i] = random_EX
                    parallel_paper_dict = replace_a_in_paper_dict(random_EX, data_dict[random_EX], i,
                                                                  parallel_paper_dict)


        else:

            jiaoji = k2e_dict[kn[0]]
            for k in kn:
                b = k2e_dict[k]
                jiaoji = list(set(jiaoji) & set(b))

            random_EX = random.choice(jiaoji)
            while True:
                if random_EX not in parallel_paper_list:
                    break
                else:

                    random_EX = random.choice(jiaoji)

            parallel_paper_list[i] = random_EX
            parallel_paper_dict = replace_a_in_paper_dict(random_EX, data_dict[random_EX], i, parallel_paper_dict)

    parallel_papers.append(parallel_paper_list)

    input_stu_ids, input_exer_ids, input_knowledge_embs = get3data(select_u_list, parallel_paper_list,
                                                                   parallel_paper_dict)
    input_stu_ids, input_exer_ids, input_knowledge_embs = input_stu_ids.to(device), input_exer_ids.to(
        device), input_knowledge_embs.to(device)

    stu4ex_scores, score_list = get_stu4ex_scores(input_stu_ids, input_exer_ids, input_knowledge_embs)
    r_parallel_12345 = get_reward(stu4ex_scores, score_list, parallel_paper_dict)

    reliability.append(score_list)

    csv_file = f'./papers/{args.data_name}/GEPG_20_parallel_papers.csv'

    headers = ['r', 'r1', 'r2', 'r3', 'r4']

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow(r_parallel_12345)


def count_common_elements(list1, list2):
    return len(set(list1) & set(list2))


if __name__ == '__main__':

    seed = args.random_seed
    local_map = construct_local_map(args)
    net = Net(args, local_map).to(device)
    model_path = f'./model/{args.data_name}/model_epoch_20.pth'
    net = load_snapshot(net, model_path, device)
    net.eval()
    data_loader = TrainDataLoader(args.data_name, args.knowledge_n, args.batch_size, True)

    QB = []
    with open(os.path.join(f"./data/{args.data_name}/", "exer.txt"), 'r') as file:
        for line in file:
            q = int(line.strip())
            QB.append(q)

    p2idx = {p: idx for idx, p in enumerate(QB)}
    idx2p = {idx: p for idx, p in enumerate(QB)}

    s_list = []
    with open(os.path.join(f"./data/{args.data_name}/", "know.txt"), 'r') as file:
        for line in file:
            s = int(line.strip())
            s_list.append(s)

    stu_num = 50
    select_u_list = get_student(stu_num)

    data_dict = get_all_dir()

    print("dataset_name:", args.data_name)

    n_states = args.q_embedding_size * 100
    n_actions = len(QB)
    v_min = 0.0
    v_max = 200.0
    atom_size = 51
    support = torch.linspace(v_min, v_max, atom_size).to(device)
    n_step = 3
    num_frames = 5000
    gamma = 0.99
    target_update = 100
    epsilon_decay = 0.001
    memory_n = PrioritizedReplay(n_states, args.memory_size, args.rl_batch_size, n_step=n_step,
                                 gamma=gamma)

    cho_prob = Rainbow(n_states, args.rl_hidden_size, n_actions, atom_size, support,
                       args.memory_size, args.rl_learning_rate, args.rl_batch_size)

    agent = DQNAgent(n_states, n_actions, cho_prob,
                     PrioritizedReplay,
                     args.memory_size, args.rl_batch_size, target_update, epsilon_decay,
                     double_dqn=False, is_noisy=False, n_step=n_step, gamma=gamma,
                     memory_n=memory_n,
                     is_categorical=False, v_min=v_min, v_max=v_max,
                     atom_size=atom_size)
    # papers
    ExamPaperGeneration(data_dict)

    # parallel papers
    reliability = []
    parallel_papers = []
    start_time = time.time()
    for j in range(20):
        choose_by_node()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time spent: {execution_time}")

    with open(f"./papers/{args.data_name}/parallel_papers.pkl", "wb") as f:
        pickle.dump(parallel_papers, f)

    minmax_scaler = MinMaxScaler()
    data_normalized = minmax_scaler.fit_transform(reliability)
    data_normalized = pd.DataFrame(data_normalized)
    r5 = pg.cronbach_alpha(data_normalized)[0]
    print(f"r5: {r5}")

    pairs = list(combinations(parallel_papers, 2))
    repeat_rate = []
    for pair in pairs:
        repeat_rate.append(count_common_elements(pair[0], pair[1]))
    print(sum(repeat_rate) / (len(parallel_papers) * 100))

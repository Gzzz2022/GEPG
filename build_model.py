import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from Graph_model.data_loader import TrainDataLoader, ValTestDataLoader
from Graph_model.dgl_graph import dgl_graph
from Graph_model.model import Net
from ctl import CommonArgParser


def train(args, local_map):
    data_loader = TrainDataLoader(args.data_name, args.knowledge_n, args.batch_size)
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

    net = Net(args, local_map)
    net = net.to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output + 1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()

            if batch_count % args.log_interval == 0:
                print(
                    f'train--{epoch + 1}/{args.epoch_n} [{batch_count}/{data_loader.get_batch_num()}] loss: {running_loss / args.log_interval}')
                running_loss = 0.0

        if not os.path.exists(os.path.join(args.model_root_save_dir, args.model_save_dir)):
            os.makedirs(os.path.join(args.model_root_save_dir, args.model_save_dir))
        save_snapshot(net, f'{args.model_root_save_dir}/{args.model_save_dir}/model_epoch_{str(epoch + 1)}.pth')

        predict(args, net, epoch)


def predict(args, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = TrainDataLoader(args.data_name, args.knowledge_n, args.batch_size, True)
    print('predicting model1...')
    data_loader.reset()
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)

        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.tolist()
        label_all += labels.tolist()

        if batch_count % args.log_interval == 0:
            print(
                f'val--{epoch + 1}/{args.epoch_n} [{batch_count}/{data_loader.get_batch_num()}] acc: {correct_count / exer_count}')

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    accuracy = correct_count / exer_count

    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))

    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))
    if not os.path.exists(os.path.join(args.result_root_save_dir, args.result_save_dir)):
        os.makedirs(os.path.join(args.model_root_save_dir, args.model_save_dir))
    with open(f'{args.result_root_save_dir}/{args.result_save_dir}/ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, accuracy, rmse, auc))


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def construct_local_map(args):
    local_map = {

        'undirected_g': dgl_graph(args.data_name, 'undirect', args.knowledge_n),
        'e_to_k': dgl_graph(args.data_name, 'e_to_k', args.knowledge_n + args.exer_n),
        'k_to_e': dgl_graph(args.data_name, 'k_to_e', args.knowledge_n + args.exer_n),
        'e_to_u': dgl_graph(args.data_name, 'e_to_u', args.student_n + args.exer_n),
        'u_to_e': dgl_graph(args.data_name, 'u_to_e', args.student_n + args.exer_n),
    }
    return local_map


def adjust_args(args):
    if args.model_save_dir is None:
        args.model_save_dir = args.data_name
    if args.result_save_dir is None:
        args.result_save_dir = args.data_name

    if args.exer_n is None and args.knowledge_n is None and args.student_n is None:
        with open(f'data/{args.data_name}/config.txt') as i_f:
            i_f.readline()
            args.student_n, args.exer_n, args.knowledge_n = list(map(eval, i_f.readline().split(',')))
    if args.exer_n is None:
        raise ValueError('exer_n is None')
    if args.knowledge_n is None:
        raise ValueError('knowledge_n is None')
    if args.student_n is None:
        raise ValueError('student_n is None')


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    adjust_args(args)
    train(args, construct_local_map(args))

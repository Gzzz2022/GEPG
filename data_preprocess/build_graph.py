import argparse
import json
import random
from typing import List


def build_local_map(data_file: str, exer_n: int, u_e: bool, save_files: List[str]):
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    print(len(data))

    exercise_to_target = set()  # e(src) to u(dst)
    target_to_exercise = set()  # u(src) to e(dst)

    for line in data:
        user_id = line['user_id']
        if u_e:
            exer_id = line['exer_id']
            exercise_to_target.add(str(exer_id) + '\t' + str(user_id + exer_n) + '\n')
            target_to_exercise.add(str(user_id + exer_n) + '\t' + str(exer_id) + '\n')
        else:
            for log in line['logs']:
                exer_id = log['exer_id']
                for k in log['knowledge_code']:
                    exercise_to_target.add(str(exer_id) + '\t' + str(k + exer_n) + '\n')
                    target_to_exercise.add(str(k + exer_n) + '\t' + str(exer_id) + '\n')
    print(len(exercise_to_target))
    with open(save_files[0], 'w') as f:
        for item in exercise_to_target:
            f.write(item)
    with open(save_files[1], 'w') as f:
        for item in target_to_exercise:
            f.write(item)


def build_u_e_graph(data_name: str, exer_n: int):
    data_file = f'../data/{data_name}/train_set.json'
    save_files = [
        f'../data/{data_name}/graph/e_to_u.txt',
        f'../data/{data_name}/graph/u_to_e.txt'
    ]
    build_local_map(data_file, exer_n, True, save_files)


def build_k_e_graph(data_name: str, exer_n: int):
    data_file = f'../data/{data_name}/log_data.json'
    save_files = [
        f'../data/{data_name}/graph/e_to_k.txt',
        f'../data/{data_name}/graph/k_to_e.txt'
    ]
    build_local_map(data_file, exer_n, False, save_files)


def build_interaction_graph(data_name: str):
    with open(f'../data/{data_name}/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    build_u_e_graph(data_name, exer_n)
    build_k_e_graph(data_name, exer_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='c')
    args = parser.parse_args()

    build_interaction_graph(args.data_name)

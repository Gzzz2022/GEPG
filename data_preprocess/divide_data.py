import argparse
import json
import random


def divide_data(data_name: str, min_log: int):
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    with open(f'../data/{data_name}/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    l_log = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        else:
            l_log += stus[stu_i]['log_num']
        stu_i += 1
    print('Number of Students: {}'.format(stu_i))
    exers = set()
    knows = set()
    logs_num = 0
    # 2. divide dataset into train_set and test_set
    train_set, test_set = [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.8)
        test_size = stu['log_num'] - train_size
        logs = []
        logs_num += stu['log_num']
        for log in stu['logs']:
            logs.append(log)
            exers.add(log['exer_id'])
            knows.update(log['knowledge_code'])
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})

        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[train_size:]
        # test_set.append(stu_test)
        for log in stu_train['logs']:
            test_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
        # shuffle logs in train_slice together, get train_set
    random.shuffle(train_set)
    random.shuffle(test_set)

    print('Number of logs: {}'.format(logs_num))
    print('Number of Exercises: {}'.format(len(exers)))
    print('Number of Knowledge Concepts: {}'.format(len(knows)))

    with open(f'../data/{data_name}/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(f'../data/{data_name}/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)  # 直接用test_set作为val_set
    with open(f'../data/{data_name}/config.txt', 'w', encoding='utf8') as output_file:
        output_file.write('# Number of Students, Number of Exercises, Number of Knowledge Concepts\n')
        output_file.write('%d, %d, %d\n' % (stu_i, len(exers), len(knows)))
    with open(f'../data/{data_name}/exer.txt', 'w', encoding='utf8') as output_file:
        for exer in exers:
            output_file.write('%s\n' % exer)
    with open(f'../data/{data_name}/know.txt', 'w', encoding='utf8') as output_file:
        for know in knows:
            output_file.write('%s\n' % know)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='c')
    parser.add_argument('--min_log', type=int, default=0)
    args = parser.parse_args()

    divide_data(args.data_name, args.min_log)

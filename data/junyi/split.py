import json

with open('./train_set.json', 'r') as f:
    data = json.load(f)

# 分割为两个文件
with open('./train_set_1.json', 'w') as f:
    json.dump(data[:len(data)//2], f)
with open('./train_set_2.json', 'w') as f:
    json.dump(data[len(data)//2:], f)

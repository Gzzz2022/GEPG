# GEPG: Student-Question-Knowledge Graph-based Reinforcement Learning Framework for Exam Paper Generation

## Prerequisites
Our experimental code is implemented in Python 3.8.0 using Pytorch 2.3.0.

## Environment Settings
dgl==2.4.0+cu121

matplotlib==3.7.2

numpy==1.24.3

pandas==1.5.3

pingouin==0.5.5

scikit-learn==1.3.0

scipy==1.10.1

torch==2.3.0

tqdm==4.66.5

To install these packages, you can use the command: 

pip install -r requirements.txt

## Dataset
junyi dataset: 
  The `log_data.json` file  is too large to be stored on GitHub, so it has been split into two files, `log_data_1.json` and `log_data_2.json`. You will need to merge them manually before use.
  The same applies to the `train_set.json`.

## Example to Run the Codes

To run the codes using the `assistment` dataset:

1. build model

```python
python build_model.py --data_name=assistment
```

2. Exam Paper Generation

```python
python main.py --data_name=assistment
```


# GEPG: Student-Question-Knowledge Graph-based Reinforcement Learning Framework for Exam Paper Generation

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


---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: description
    dtype: string
  - name: category
    dtype: string
  - name: close_time
    dtype: string
  - name: ground_truth
    dtype: string
  - name: market_probability
    dtype: 'null'
  - name: series_ticker
    dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_bytes: 733525
    num_examples: 1531
  download_size: 202676
  dataset_size: 733525
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/LOG/realman/LLM/RT1datasets/fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor-train.array_record-00000-of-01024"
index_path = None
description = {"image": "byte", "label": "float"}
dataset = TFRecordDataset(tfrecord_path, index_path, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)
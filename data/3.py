# 该文件是2023.07.06请教陈文轩时做的数据集的一些测试，能够读取文件，但是不能打印出相关信息,😴蓝瘦香菇
import tensorflow_datasets as tfds
import sys
sys.setrecursionlimit(100000)

# dataset_name = "/LOG/realman/LLM/RT1datasets/fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor"
dataset_name = "/LOG/realman/LLM/RT1datasets"
# dataload = tfds.load(dataset_name)
dataload = tfds.builder_from_directory(dataset_name)
# dataset_ex = dataload.as_dataset(split='train[:10%]')

# print(dataload)
# print(type(dataload))
dataset = dataload.as_data_source(split='train[:10%]')
# dataset_ex = tfds.as_dataset(dataset)
# print(type(dataset_ex))
# dataset = dataset[:10]
# (split='train')
# dataset = dataload.dataset_info_from_configs()
index = [0,1,2,3]
# print(dataset._read_instructions())
print(type(dataset))
print(dataset)
# for key in dataset:
#     print(key)
# print(dataset['train'])


# set(split='train')
# for example in dataset:
#     image = example['image']
#     # label = example['label']
#     print('Image shape: ', image.shape)
#     # print('Label: ', label)
    
#     # print(example)

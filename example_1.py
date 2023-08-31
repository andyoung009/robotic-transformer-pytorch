import torch
# 因为from tokenizers import AddedToken，ImportError: cannot import name 'AddedToken' from 'tokenizers' 
# 发现存在2个tokenizers，一个为当前项目路径下文件如下打印信息，另一个为安装的库（和transformers搭配的）
import sys
print(sys.path)

import os
sys.path.append('/LOG/realman/LLM/robotic-transformer-pytorch')
sys.path.remove('/LOG/realman/LLM/robotics_transformer')
print(sys.path)

unwanted_folder_path = '/LOG/realman/LLM/robotics_transformer/tokenizers'
if unwanted_folder_path in sys.path:
    sys.path.remove(unwanted_folder_path)
print(sys.path)

import tokenizers
print('**********************')
print(tokenizers.__file__)
print('**********************')

from transformers import EfficientNetConfig, EfficientNetModel
from robotic_transformer_pytorch import MaxViT, RT1

# 尝试在GPU运行代码并将use_attn_conditioner=True后发现报错，报错原因是
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
video = torch.randn(2, 3, 224, 224)
# .to(device)
vit = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 32,
    depth = (2, 2, 5, 2),
    window_size = 7,
    mbconv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)
# .to(device)
out_of_vit = vit(video)
print(out_of_vit.shape)

configuration = EfficientNetConfig()
efficient_model = EfficientNetModel(configuration)
configuration = efficient_model.config


model = RT1(
    vit = vit,
    num_actions = 11,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2,
    # use_attn_conditioner=True
)
# .to(device)

video = torch.randn(2, 3, 6, 224, 224)
# .to(device)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]
instructions_2 = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

train_logits = model(video, instructions) # (2, 6, 11, 256) # (batch, frames, actions, bins)

# after much training

model.eval()
eval_logits = model(video, instructions, cond_scale = 3.) # classifier free guidance with conditional scale of 3
eval_logits_2 = model(video, instructions_2, cond_scale = 3.)
print(eval_logits==eval_logits_2)
print(eval_logits)
print(eval_logits.shape)
# ------------------------------------------------------------------------
# Robotic-transformer-torch
# Copyright (c) 2023 andyoung. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from tokenlearner-pytorch (https://github.com/rish-16/tokenlearner-pytorch)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        B, H, W, C = x.shape
        # x = x.view(B, C, H, W)
        # 上句代码有问题，修改如下
        x = x.permute(0,3,1,2)
                
        # 获取x在第一个维度（通道）最大值并添加维度1（原来的通道维度），mx.shape = (B, 1, H, W)
        mx = torch.max(x, 1)[0].unsqueeze(1)
        # 输入x的第1维度上计算平均值并添加维度1（原来的通道维度）, avg.shape = (B, 1, H, W)
        avg = torch.mean(x, 1).unsqueeze(1)
        # 特征拼接，包含最大值信息和平均信息，在维度1拼接（原来的通道维度）
        combined = torch.cat([mx, avg], dim=1)  # combined.shape = (B, 2, H, W)
        fmap = self.conv(combined)  # fmap.shape = (B, 1, H, W)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))   # out.shape = (B, C), x * weight_map = (B, C, H, W)
        
        return out, x * weight_map

class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        B, _, _, C = x.shape
        # # 实现1，源代码写法
        # Z = torch.Tensor(B, self.S, C)
        # for i in range(self.S):
        #     Ai, _ = self.tokenizers[i](x) # [B, C]
        #     Z[:, i, :] = Ai # Z.shape = (B, self.S, C)
        
        # 实现2，有利于并行化
        Z = torch.stack([self.tokenizers[i](x)[0] for i in range(self.S)], dim=1)  # Z.shape = (B, self.S, C)
        return Z

# 对于RT-1-torch实际上并没有利用一下部分 
class TokenFuser(nn.Module):
    def __init__(self, H, W, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S
        
    def forward(self, y, x):
        B, S, C = y.shape
        B, H, W, C = x.shape
        
        Y = self.projection(y.view(B, C, S)).view(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).view(B, H*W, S) # [B, HW, S]
        BwY = torch.matmul(Bw, Y)
        
        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H*W, C)
        
        out = (BwY + xj).view(B, H, W, C)
        
        return out 


if __name__ == '__main__':
    tklr = TokenLearner(S=8)
    x = torch.rand(6, 32, 32, 512)
    y = tklr(x) # torch.Size([6, 8, 512])
    print(y.shape)
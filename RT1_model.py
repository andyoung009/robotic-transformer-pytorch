import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from film_efficientnet_pytorch.model import EfficientNet, MBConvBlock
from film_efficientnet_pytorch.utils import get_model_params, MemoryEfficientSwish
from film_efficientnet_pytorch.USE import USEncoder
from film_efficientnet_pytorch.film_efficient_model import FiLM, FiLMBlock, FiLMEfficientNet
from film_efficientnet_pytorch.tokenlearner_pytorch import TokenLearner
from film_efficientnet_pytorch.transformer_decoder import Transformers_Decoder

"""
RT-1 model architecture class
"""
class RT1model(nn.Module):
    def __init__(
        self,
        # USEncoder,
        # backbone,
        # film, # film可以不作为传递参数
        # tokenlearner,
        # transformers_decoder,
        num_classes=1000
    ):
        super().__init__()
        
        # define the USEncoder
        self.USEncoder = USEncoder()
        
        # Load EfficientNet backbone with pre-trained weights, copy the weight to the backbone model
        self.pretrained_backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = self.pretrained_backbone
        self.backbone._blocks_args, self.backbone._global_params = get_model_params('efficientnet-b3', None)
        self.backbone = EfficientNet(blocks_args=self.backbone._blocks_args, global_params=self.backbone._global_params)
        self.backbone.load_state_dict(self.pretrained_backbone.state_dict())

        # self._swish = MemoryEfficientSwish()
        
        # self.film = FiLM(self.USEncoder._hidden_size, out_channels)
        self.backbone_with_film = []

        # Replace or append MBConvBlock with FiLMBlock, 添加到一个新的modulelist中去
        for idx, block in enumerate(self.backbone._blocks):
            # self.backbone._blocks[idx] = FiLMBlock(block)
            self.backbone_with_film.append(block)
            self.backbone_with_film.append(FiLM(self.USEncoder._hidden_size, block._bn2.num_features))
        
        self.backbone_with_film = nn.ModuleList(self.backbone_with_film)

        self.Linear_1b1_conv = nn.Conv2d(1536, 512, 1)

        self.tokenlearner = TokenLearner(S=8)

        self.transformers_decoder = Transformers_Decoder(dim=512, depth=8, heads=8, dim_head=64, mlp_dim=512, dropout = 0., d_model = 512, max_seq_len = 48, num_actions = 11, vocab_size = 256)
        # Replace the last linear layer with a new one
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x, context_sentences):
        # context换成sentences，然后把相关的处理放到前向函数中来！
        context = self.USEncoder(context_sentences)
        # Stem
        # x = inputs
        inputs = x
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(inputs)))

        for idx, block in enumerate(self.backbone_with_film):
            if isinstance(block, MBConvBlock):
            # if 'MBConv' in block.name:
                x = block(x)
            elif isinstance(block, FiLM):
            # elif 'FiLM' in block.name:
                x = block(x, context)
            else:
                assert True, ' The block type should be MBConv or FiLM. ' 

            # x = self.backbone._blocks(x, context)
        x = self.backbone._swish(self.backbone._bn1(self.backbone._conv_head(x)))
        # 添加通道转换的卷积模块，从efficientnet最后的1536通道转为512通道
        x = self.Linear_1b1_conv(x)

        x = x.permute(0,2,3,1)
        x = self.tokenlearner(x)
        x = self.transformers_decoder(x)
        # 原来的模型尾部处理注释掉，主要是为了提取融合instruction的特征
        # x = self.backbone.extract_features(x)  # Get features from the backbone
        # x = self.backbone._avg_pooling(x)  # Global average pooling
        # x = x.flatten(start_dim=1)  # Flatten
        # x = self.classifier(x)  # Classification layer
        return x

if __name__ == '__main__':
    input_tensor = torch.randn(6,3,300,300)
    sentences = ["Pick apple from top drawer and place on counter."]
    model = RT1model()
    output_tensor = model(input_tensor, sentences)
    print(output_tensor.shape)
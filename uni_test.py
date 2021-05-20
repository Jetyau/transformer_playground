import torch
import pdb
import transformer
from transformer import VisionTransformer, VisionTransformerUpHead

from transformer import config

backbone_arg = config.model['backbone']
decode_head_arg = config.model['decode_head']
pdb.set_trace()
model = VisionTransformer(model_name='vit_large_patch16_384', **backbone_arg).to('cuda:1')
decoder_head = VisionTransformerUpHead(**decode_head_arg).to('cuda:1')
tmp = torch.rand((1,3,256,256), device='cuda:1')
output = model(tmp)
output_2 = decoder_head(output)
print(output[-1].shape)
print(output_2.shape)
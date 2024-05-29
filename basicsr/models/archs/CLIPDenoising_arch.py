

import torch
from torch import nn

from basicsr.models.archs.CLIPEncoder_util import (ModifiedResNet,
                                                      UNetUpBlock,
                                                      UNetUpBlock_nocat,
                                                      conv3x3)

class ModifiedResNet_RemoveFinalLayer(ModifiedResNet):
   
    def __init__(self, layers, in_chn=3, width=64):
        super().__init__(layers, in_chn, width)

    def forward(self, x):
        out = []

        x = x.type(self.conv1.weight.dtype); out.append(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); out.append(x)
        x = self.avgpool(x)
        
        x = self.layer1(x); out.append(x)
        x = self.layer2(x); out.append(x)
        x = self.layer3(x); out.append(x)
        x = self.layer4(x); 

        return out

class CLIPDenoising(nn.Module):
    def __init__(self, num_blocks, inp_channels=3, out_channels=3, depth=5, wf=64, slope=0.2,
                       bias=True, model_path=None, aug_level=0.05):

        super(CLIPDenoising, self).__init__()
        
        self.sigmas = [aug_level * i for i in range(1, 5)]
        
        self.inp_channels = inp_channels
        if inp_channels == 1: # used for 1 channel input, eg. CT images
            self.first = nn.Conv2d(inp_channels, 3, kernel_size=1, bias=bias)
            inp_channels = 3

        self.encoder = ModifiedResNet_RemoveFinalLayer(num_blocks, inp_channels, width=wf)
        self.encoder.load_pretrain_model(model_path)
            
        for params in self.encoder.parameters():
            params.requires_grad = False

        # learnable decoder
        self.up_path = nn.ModuleList()
        prev_channels = wf * 2 ** (len(num_blocks))
        for i in range(depth):
            if i == 0:
                self.up_path.append(UNetUpBlock_nocat(prev_channels, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 2:
                self.up_path.append(UNetUpBlock(prev_channels*3//2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2
            elif i == depth - 1: # introduce noisy image as a dense feature
                self.up_path.append(UNetUpBlock(prev_channels+inp_channels, prev_channels, slope, bias))
            else:
                self.up_path.append(UNetUpBlock(prev_channels*2, prev_channels//2, slope, bias))
                prev_channels = prev_channels//2

        self.last = conv3x3(prev_channels, out_channels, bias=bias)


    def forward(self, x):
        
        if self.inp_channels == 1:
            x = self.first(x)
            
        out = self.encoder(x)

        # progressive feature augmentation 
        if self.training:
            for idx in range(len(out)):
                if idx == 0: continue # 
                alpha = torch.randn_like(out[idx]) * self.sigmas[idx-1] + 1.0
                out[idx] = out[idx] * alpha
                
        x = out[-1]
         
        for i, up in enumerate(self.up_path):
            if i != 0: 
                x = up(x, out[-i-1])
            else:
                x = up(x)

        return self.last(x)
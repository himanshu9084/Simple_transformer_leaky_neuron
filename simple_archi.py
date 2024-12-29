from spikingjelly_codes.activation_based import functional, layer, surrogate, neuron
from spikingjelly_codes.activation_based.neuron import BaseNode, LIFNode
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from set_seed import _seed_
import torch
import math
import os
import random 

random.seed(_seed_)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(_seed_)


###################################### Simple_transformer ############################################
##########################################################################################################


from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
from spikingjelly_codes.activation_based.neuron import BaseNode, LIFNode
from torchvision import transforms



class PLIFNode(BaseNode):
    ## code for PLIF taken from https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/tree/main
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=surrogate.ATan(), monitor_state=False, step_mode='s'):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            # self.v += dv - self.v * self.w.sigmoid()
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        #return self.spiking()
        return self.neuronal_fire()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'



class Resnet_extractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(Resnet_extractor, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        #self.backbone = models.resnet50(pretrained=pretrained)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Removes classifier layer
        
        # Modified conv 1 layer for 2-channel input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,  
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
    def forward(self, x):
        return self.backbone(x)

    
class ResidualSpikingLayer(nn.Module):
    def __init__(self, input_dim):
        super(ResidualSpikingLayer, self).__init__()
        #self.lif1 = neuron.LIFNode(tau=2.0)
        self.lif1 = PLIFNode(init_tau=2.0)
        self.fc = nn.Linear(input_dim, input_dim)
        #self.lif2 = neuron.LIFNode(tau=2.0)
        self.lif2 =PLIFNode(init_tau=2.0)

    def forward(self, x):
        residual = x
        x = self.lif1(x)
        x = self.fc(x)
        x = self.lif2(x) + residual
        return x    
    
    
class temporal_processor(nn.Module):
    def __init__(self, input_dim, hidden_dim, T=20):
        super(temporal_processor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.lif = neuron.LIFNode(tau=2.0)
        #self.lif = PLIFNode(init_tau=2.0)
        self.lif = ResidualSpikingLayer(hidden_dim) # For residual spiking layers
        self.T = T  # Number of timesteps

    def forward(self, x):
        # x - [batch_size, T, feature_dim]
        outputs = []
        for t in range(self.T):
            out = self.fc1(x[:, t, :])
            out = self.lif(out)
            outputs.append(out)
        return torch.stack(outputs, dim=1)
    
class Simple_Transformer(nn.Module):
    def __init__(self, hidden_dim, nhead, num_layers, seq_length):
        super(Simple_Transformer, self).__init__()
        self.encoder1 = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.encoder2 = TransformerEncoder(self.encoder1, num_layers=num_layers)
        self.seq_length = seq_length

    def forward(self, x):
        # x: [batch_size, T, hidden_dim]
        x = x.permute(1, 0, 2)  # [T, batch_size, hidden_dim]
        x = self.encoder2(x)  # [T, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)  # [batch_size, T, hidden_dim]
        return x
    
    
class Simple_SNN(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, hidden_dim=256, nhead=4, num_layers=2, T=20, num_classes=11):
        super(Simple_SNN, self).__init__()
        self.feature_extractor = Resnet_extractor(backbone, pretrained)
        
        self.temporal_processor = temporal_processor(self.feature_extractor.feature_dim, hidden_dim, T)
        self.spiking_transformer = Simple_Transformer(hidden_dim, nhead, num_layers, seq_length=T)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, T, channels, height, width = x.size()
        x = x.view(batch_size * T, channels, height, width)  # Flatten the time dimension
        features = self.feature_extractor(x)
        features = features.view(batch_size, T, -1)  # Reshape for temporal processing
        temporal_out = self.temporal_processor(features)
        transformer_out = self.spiking_transformer(temporal_out)
        out = torch.mean(transformer_out, dim=1)  # average across time
        out = self.fc(out)
        return out

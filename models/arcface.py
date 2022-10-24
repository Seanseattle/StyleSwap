
from collections import namedtuple
import math
from PIL import Image
import numpy as np

import paddle.nn as nn
import paddle
import paddle.nn.functional as F


class Flatten(nn.Layer):
    def forward(self, input):
        return input.reshape([input.shape[0], -1])


class SEModule(nn.Layer):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(
            channels, channels // reduction, kernel_size=1, padding=0, bias_attr=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(
            channels // reduction, channels, kernel_size=1, padding=0, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(nn.Layer):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2D(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False), nn.BatchNorm2D(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2D(in_channel),
            nn.Conv2D(in_channel, depth, (3, 3), (1, 1), 1 ,bias_attr=False), nn.PReLU(depth),
            nn.Conv2D(depth, depth, (3, 3), stride, 1 ,bias_attr=False), nn.BatchNorm2D(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(nn.Layer):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2D(1, stride)
        else:
            self.shortcut_layer =nn.Sequential(
                nn.Conv2D(in_channel, depth, (1, 1), stride ,bias_attr=False),
                nn.BatchNorm2D(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2D(in_channel),
            nn.Conv2D(in_channel, depth, (3,3), (1,1),1, bias_attr=False),
            nn.PReLU(depth),
            nn.Conv2D(depth, depth, (3,3), stride, 1, bias_attr=False),
            nn.BatchNorm2D(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class MapperLayer(nn.Layer):
    def __init__(self, inc=512, out_c=512, width=7, drop_ratio=0.4):
        super(MapperLayer, self).__init__()
        self.output_layer2 = nn.Sequential(nn.BatchNorm2D(inc),
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(inc * width * width, out_c),
                                       nn.BatchNorm1D(out_c))
    def load_weights(self, model):
        self.output_layer2.load_state_dict(model.state_dict())

    def forward(self, x):
        x = self.output_layer2(x)
        return x
        
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


class Backbone(nn.Layer):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2D(3, 64, (3, 3), 1, 1, bias_attr=False),
                                      nn.BatchNorm2D(64),
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2D(512),
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(512 * 7 * 7, 512),
                                       nn.BatchNorm1D(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x, theta):
        """
        foward
        """
        if theta is not None:
            grid = F.affine_grid(theta, (x.shape[0], 3, 224, 224), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, mode='bilinear')
        if x.shape[-1] != 112:
            x = F.interpolate(x, (112, 112))
        x = self.input_layer(x)
        id_emb = self.body(x)
        x = self.output_layer(id_emb)
        return x

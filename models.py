import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.transforms._presets import ImageClassification
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models.resnet import ResNet18_Weights

from typing import Any, Callable, List, Optional, Type, Union, Tuple

class FCNN(nn.Module):

  def __init__(self, input_size, num_hidden, num_classes):
    super(FCNN, self).__init__()
    self.fc1 = nn.Linear(input_size, num_hidden)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(num_hidden, num_classes)

  def forward(self, x):

    y = self.fc1(x)
    y = self.relu(y)
    y = self.fc2(y)

    return y
    
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(3, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def make_resnet18k(k=64, num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k)


def get_model(model, dataset, num_hidden_units=1, width=64):

    if model.lower() == 'fcnn':
        
        if dataset.lower() == 'mnist':
            input_size = 784
            num_classes = 10

        return FCNN(input_size, num_hidden_units, num_classes)
    
    if model.lower() == 'resnet18':
        
        if dataset.lower() == 'cifar10':
            input_size = (224, 224, 3)
            num_classes = 10
        elif dataset.lower() == 'cifar100':
            input_size = (224, 224, 3)
            num_classes = 100
        
        return make_resnet18k(k = width, num_classes = num_classes)  
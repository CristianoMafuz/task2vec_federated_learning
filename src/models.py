# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import torch
import torchvision.models.resnet as resnet
from torchvision.models import resnet34 as tv_resnet34, ResNet34_Weights

from task2vec import ProbeNetwork


_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn


class ResNet(resnet.ResNet, ProbeNetwork):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.layers = [
            self.conv1, self.bn1, self.relu,
            self.maxpool, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
            lambda z: torch.flatten(z, 1), self.fc
        ]

    @property
    def classifier(self):
        return self.fc

    def forward(self, x, start_from=0):
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x


@_add_model
def resnet34(pretrained=False, num_classes=1000):
    # Usa a classe ResNet customizada (com .layers)
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    if pretrained:
        from torchvision.models import ResNet34_Weights, resnet34 as tv_resnet34
        # Carrega pesos do torchvision e transfere para o modelo customizado
        state_dict = tv_resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)

    return model



def get_model(model_name, pretrained=False, num_classes=1000):
    try:
        return _MODELS[model_name](pretrained=pretrained, num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Architecture {model_name} not implemented.")


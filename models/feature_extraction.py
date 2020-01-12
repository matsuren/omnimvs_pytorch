import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes=32, planes=32, dilate=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, 1, dilate, dilate, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, dilate, dilate, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu2(out)
        return out


class UnaryExtraction(nn.Module):
    def __init__(self, input_channel=1):
        super(UnaryExtraction, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.ReLU(inplace=True))
        # conv2_11
        self.conv2_11 = self._make_layer(BasicBlock, 5)
        # conv12_17 with dilate
        dilates = [2, 3, 4]
        self.conv12_17 = self._make_layer(BasicBlock, len(dilates), dilates)

    def _make_layer(self, block, blocks, dilates=None):
        if dilates is None:
            dilates = [1 for _ in range(blocks)]
        else:
            assert len(dilates) == blocks
        #
        layers = []
        for i in range(blocks):
            layers.append(block(dilate=dilates[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_11(out)
        out = self.conv12_17(out)
        return out

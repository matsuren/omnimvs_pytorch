import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3d_relu(inplanes, planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=True),
        nn.ReLU(inplace=True))


class CostVolumeComputation(nn.Module):
    def __init__(self, inplanes=64):
        super(CostVolumeComputation, self).__init__()
        # conv
        planes = 64
        self.conv1 = conv_3d_relu(inplanes, planes, 3, 1, 1)
        self.conv23 = self._make_layer(conv_3d_relu, planes, 2)

        planes = 64 * 2
        self.conv4 = conv_3d_relu(planes // 2, planes, 3, 2, 1)
        self.conv56 = self._make_layer(conv_3d_relu, planes, 2)
        self.conv7 = conv_3d_relu(planes, planes, 3, 2, 1)
        self.conv89 = self._make_layer(conv_3d_relu, planes, 2)
        self.conv10 = conv_3d_relu(planes, planes, 3, 2, 1)
        self.conv11_12 = self._make_layer(conv_3d_relu, planes, 2)

        planes = 128 * 2
        self.conv13 = conv_3d_relu(planes // 2, planes, 3, 2, 1)
        self.conv14_15 = self._make_layer(conv_3d_relu, planes, 2)

        # deconv
        planes = 128
        self.deconv1 = nn.ConvTranspose3d(planes * 2, planes, 3, 2, 1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose3d(planes, planes, 3, 2, 1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose3d(planes, planes, 3, 2, 1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        planes = 64
        self.deconv4 = nn.ConvTranspose3d(planes * 2, planes, 3, 2, 1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose3d(planes, 1, 3, 2, 1, bias=True)

    def _make_layer(self, block, planes, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(block(planes, planes, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # conv
        conv1 = self.conv1(x)
        conv3 = self.conv23(conv1)
        conv4 = self.conv4(conv1)
        conv6 = self.conv56(conv4)
        conv7 = self.conv7(conv4)
        conv9 = self.conv89(conv7)
        conv10 = self.conv10(conv7)
        conv12 = self.conv11_12(conv10)
        conv13 = self.conv13(conv10)
        conv15 = self.conv14_15(conv13)

        # deconv
        deconv1 = self.deconv1(conv15, output_size=conv12.size())
        deconv1 = self.relu1(deconv1 + conv12)
        deconv2 = self.deconv2(deconv1, output_size=conv9.size())
        deconv2 = self.relu2(deconv2 + conv9)
        deconv3 = self.deconv3(deconv2, output_size=conv6.size())
        deconv3 = self.relu3(deconv3 + conv6)
        deconv4 = self.deconv4(deconv3, output_size=conv3.size())
        deconv4 = self.relu4(deconv4 + conv3)

        # final upsample
        d, h, w = deconv4.size()[2:]
        output_size = torch.Size([2 * d, 2 * h, 2 * w])
        deconv5 = self.deconv5(deconv4, output_size=output_size)
        return deconv5

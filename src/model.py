import torchvision
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def hard_negative_mining(pred, target, weight=None):

    """
    Online hard mining on the entire batch

    :param pred: predicted character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    :param target: target character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    :param weight: If weight is not None, it denotes the weight given to each pixel for weak-supervision training
    :return: Online Hard Negative Mining loss
    """

    cpu_target = target.data.cpu().numpy()
    all_loss = F.mse_loss(pred, target, reduction='none')

    positive = np.where(cpu_target != 0)[0]
    negative = np.where(cpu_target == 0)[0]

    if weight is not None:
        positive_loss = all_loss[positive]*weight[positive]
    else:
        positive_loss = all_loss[positive]

    negative_loss = all_loss[negative]

    negative_loss_cpu = np.argsort(
        -negative_loss.data.cpu().numpy())[0:min(min(1000, 4 * positive_loss.shape[0]), negative_loss.shape[0])]

    return (positive_loss.sum() + negative_loss[negative_loss_cpu].sum()) / (
                positive_loss.shape[0] + negative_loss_cpu.shape[0])


class Criterian(nn.Module):

    def __init__(self):

        """
        Class which implements weighted OHNM with loss function being MSE Loss
        """

        super(Criterian, self).__init__()

    def forward(self, output, character_map, affinity_map, character_weight=None, affinity_weight=None):

        """

        :param output: prediction output of the model of shape [batch_size, 2, height, width]
        :param character_map: target character map of shape [batch_size, height, width]
        :param affinity_map: target affinity map of shape [batch_size, height, width]
        :param character_weight: weight given to each pixel using weak-supervision for characters
        :param affinity_weight: weight given to each pixel using weak-supervision for affinity
        :return: loss containing loss of character heat map and affinity heat map reconstruction
        """

        batch_size, channels, height, width = output.shape

        output = output.permute(0, 2, 3, 1).contiguous().view([batch_size * height * width, channels])

        character = output[:, 0]
        affinity = output[:, 1]

        affinity_map = affinity_map.view([batch_size * height * width])
        character_map = character_map.view([batch_size * height * width])

        if character_weight is not None:
            character_weight = character_weight.view([batch_size * height * width])

        if affinity_weight is not None:
            affinity_weight = affinity_weight.view([batch_size * height * width])

        loss_character = hard_negative_mining(character, character_map, character_weight)
        loss_affinity = hard_negative_mining(affinity, affinity_map, affinity_weight)

        all_loss = loss_character + loss_affinity

        return all_loss


class ConvBlock(nn.Module):

    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_non_linearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.with_non_linearity = with_non_linearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_non_linearity:
            x = self.ReLU(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: up-sampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):

    """
        U architecture model used for prediction. This is different from the model used in the original paper
    """

    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
                if len(down_blocks) == 2:
                    break
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 1024)
        up_blocks.append(UpBlockForUNetWithResNet50(512 + 256, 256, up_conv_in_channels=1024, up_conv_out_channels=512))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=256 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        self.final_out = nn.Sigmoid()

    def forward(self, x, with_output_feature_map=False):

        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 3):
                break
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 3 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        x = self.final_out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

import numpy as np
from torch.nn import functional as F
from torch import nn

THRESHOLD_POSITIVE = 0.3
THRESHOLD_NEGATIVE = 0.1


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

    if weight is not None:
        cpu_weight = weight.data.cpu().numpy()
        positive = np.where(np.logical_and(cpu_target >= THRESHOLD_POSITIVE, cpu_weight != 0))[0]
    else:
        positive = np.where(cpu_target >= THRESHOLD_POSITIVE)[0]

    negative = np.where(cpu_target < THRESHOLD_NEGATIVE)[0]

    if weight is not None:
        positive_loss = all_loss[positive]*weight[positive]
    else:
        positive_loss = all_loss[positive]

    negative_loss = all_loss[negative]

    negative_loss_cpu = np.argsort(
        -negative_loss.data.cpu().numpy())[0:min(max(1000, 3 * positive_loss.shape[0]), negative_loss.shape[0])]

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

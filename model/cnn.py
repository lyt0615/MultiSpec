'''
4 convs & 4 fcs, no dropout
'''


import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CNN_1(nn.Module):
    def __init__(self, class_num=37, init_weights=True):

        self.kernel_size = 3

        super(CNN_1, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.kernel_size)
        
        # Define the fully connected layers
        # self.fc1 = nn.Linear(15*1024, 4800)
        self.fc1 = nn.Linear(13824, 4800)
        self.fc2 = nn.Linear(4800, 3200)
        self.fc3 = nn.Linear(3200, 1600)
        self.fc4 = nn.Linear(1600, class_num)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        # Define activation functions and pooling layer
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout()
        
        # Define the output activation function (e.g., sigmoid for binary classification)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the tensor before passing it through fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc4(x)

        # x = self.sigmoid(x)  
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
import torch
import torch.nn.functional as F

from torchvision.utils import _log_api_usage_once


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # if all(alpha) >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

    
class FocalLoss(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets, alpha=None):
        N = inputs.size(0)
        C = inputs.size(1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.type(torch.int64)
        class_mask.scatter_(1, ids.data, 1.)

        if alpha is None:
            self.alpha = Variable(torch.ones(inputs.shape[1], 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (inputs*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))+log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class WeightedBCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedBCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):

        # input = Variable(input, requires_grad=True)
        # target = Variable(target, requires_grad=False)
        # if weight is not None:
        #     # weights_tensor = Variable(torch.stack([weights]*input.shape[0], dim=0), requires_grad=False)
        #     loss = self.BCELoss(input, target, weight)
        #     loss_mean = torch.sum(torch.sum(loss, dim=1))/input.shape[0]/input.shape[1]
        #     return loss_mean

        if weight is None: self.weight = Variable(torch.ones(input.shape[1]))
        else: self.weight = Variable(weight)

        weight_tensor = torch.stack([self.weight]*len(input))

        if input.is_cuda and not self.weight.is_cuda:
            self.weight = self.weight.cuda()

        # Apply weight to the loss of positive instances only:
        loss_positive = weight_tensor*target*(input.log())
        loss_negative = (1-target)*((1-input).log())
        loss = -loss_positive-loss_negative

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
 
    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
 
    def forward(self, input, target, weight):
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index) 
    
    
class EarlyStopping:
# Early stops the training if validation loss doesn't improve after a given patience.
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path: Te path to save model.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# save so-far-the-best parameters
        print('Model saved.')
        self.val_loss_min = val_loss

if __name__ == "__main__":
    input = torch.randn((16,1024))
    net = CNN_1()
    output = net(input)
    print(output.shape)
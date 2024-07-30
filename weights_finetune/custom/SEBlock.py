import torch
import torch.nn as nn

class SEBlock(nn.Module):
  def __init__(self, in_channels, reduction_ratio=4):
    super(SEBlock, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Sequential(
        nn.Linear(in_channels, in_channels * reduction_ratio),
        nn.ReLU(inplace=True),
        nn.Linear(in_channels * reduction_ratio, in_channels),
        nn.Sigmoid()
    )

  def forward(self, x):
    # Squeeze operation
    squeeze = self.avg_pool(x)

    # Excitation operation
    excitation = self.fc(squeeze.view(x.size(0), -1))

    # Scale operation
    scale = excitation.view(x.size(0), x.size(1), 1, 1)
    return x * scale
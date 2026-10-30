import torch
from torch import nn

class BCEDiceLoss(nn.Module):
    """BCE with positive-class weighting + soft Dice loss for sparse fault labels."""
    def __init__(self, pos_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.dice_weight=dice_weight

    def forward(self, logits, target):
        bce=nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=self.pos_weight)
        prob=torch.sigmoid(logits)
        dims=(1,2,3)
        inter=(prob*target).sum(dims)
        denom=prob.sum(dims)+target.sum(dims)
        dice=1-((2*inter+1e-6)/(denom+1e-6)).mean()
        return bce+self.dice_weight*dice

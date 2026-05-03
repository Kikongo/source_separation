import numpy as np
import torch.nn as nn

def loss_function(y_true, y_pred, alpha=1.0)->float:
      '''
      this returns the customised loss function as mentioned in the paper
      the result will be calculated as loss(Singing Voice) = alpha ∗ L(vocal, channelvocal) +(1 − alpha) ∗ L(acc, channelacc)
      where L() is the average of L1 losses on every pixel. with alpha = 1.0, we only get the vocals and not the accompaniments
      '''
      l1_loss = nn.L1Loss()
      
      vocal_loss = l1_loss(y_pred[:, 0].unsqueeze(1),  y_true[:, 0].unsqueeze(1))
      acc_loss = l1_loss(y_pred[:, 1].unsqueeze(1), y_true[:, 1].unsqueeze(1))


      #L1 loss function: https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions
      return alpha * vocal_loss + (1-alpha) * acc_loss
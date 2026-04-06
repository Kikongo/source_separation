import numpy as np
import torch.nn as nn

def loss_function(y_true:np.array, y_pred:np.array, alpha:np.array=[0.25, 0.25, 0.25, 0.25])->float:
      '''
      this returns the customised loss function as mentioned in the paper
      the result will be calculated as loss(Singing Voice) = alpha ∗ L(vocal, channelvocal) +(1 − alpha) ∗ L(acc, channelacc)
      where L() is the average of L1 losses on every pixel. with alpha = 1.0, we only get the vocals and not the accompaniments
      '''
      l1_loss = nn.L1Loss()
      
      vocal_loss = l1_loss(y_pred[:, 0, :, :] - y_true[:, 0, :, :])
      drum_loss = l1_loss(y_pred[:, 1, :, :] - y_true[:, 1, :, :])
      bass_loss = l1_loss(y_pred[:, 2, :, :] - y_true[:, 2, :, :])
      other_loss = l1_loss(y_pred[:, 3, :, :] - y_true[:, 3, :, :])

      #L1 loss function: https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions
      return alpha[0] * vocal_loss + alpha[1] * drum_loss + alpha[2] * bass_loss + alpha[3] * other_loss
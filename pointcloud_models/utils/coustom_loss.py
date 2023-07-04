import torch
import torch.nn as nn


class fit_loss(nn.Module):
    def __init__(self):
        super(fit_loss, self).__init__()

    def forward(self, pred, target):
        target.squeeze_()
        pred_xy = pred[:, 0:2]
        target_xy = target[:, 1:3]
        distance_loss = torch.mean(torch.sqrt(torch.sum((pred_xy - target_xy) ** 2, dim=1)))
        
        pred_radius = pred[:, 2]
        target_radius = target[:, 3]
        radius_loss = torch.mean(torch.abs(pred_radius - target_radius))
        
        pred_charge = pred[:, 3]
        pred_charge = torch.sigmoid(pred_charge)
        pred_charge = torch.where(pred_charge > 0.5, torch.ones_like(pred_charge), torch.zeros_like(pred_charge))

        
        target_charge = target[:, 4]
        target_charge = torch.where(target_charge > 0, torch.ones_like(target_charge), torch.zeros_like(target_charge))
        target_charge = target_charge.type(torch.float32)
        
        BinaryCrossEntropy = nn.BCELoss()
        charge_loss = BinaryCrossEntropy(pred_charge, target_charge)
        
        loss = distance_loss + radius_loss + charge_loss
        
        print('distance_loss: ', distance_loss.item(), 'radius_loss: ', radius_loss.item(), 'charge_loss: ', charge_loss.item())
        
        return loss
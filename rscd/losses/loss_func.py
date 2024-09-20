import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(CELoss, self).__init__()

        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        if not reduction:
            print("disabled the reduction.")
    
    def forward(self, pred, target):
        loss = self.criterion(pred, target) 
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class dice_loss(nn.Module):
    def __init__(self, eps=1e-7):
        super(dice_loss, self).__init__()
        self.eps = eps
    
    def forward(self, logits, true):
        """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            p = torch.eye(num_classes).cuda()
            true_1_hot = p[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class BCEDICE_loss(nn.Module):
    def __init__(self):
        super(BCEDICE_loss, self).__init__()
        self.bce = torch.nn.BCELoss()
    
    def forward(self, target, true):
        
        bce_loss = self.bce(target, true.float())

        true_u = true.unsqueeze(1)
        target_u = target.unsqueeze(1)

        inter = (true * target).sum()
        eps = 1e-7
        dice_loss = (2 * inter + eps) / (true.sum() + target.sum() + eps)

        return bce_loss + 1 - dice_loss

if __name__ == "__main__":
    predict = torch.randn(4, 2, 10, 10)
    target = torch.randint(low=0,high=2,size=[4, 10, 10])
    func = CELoss()
    loss = func(predict, target)
    print(loss)

import torch
import torch.nn as nn
from collections import deque
    

class LossFunction(nn.Module):
    """
    Composite loss function with support for:
    - standard cross-entropy loss
    - target class cross-entropy (for targeted attacks)
    - optional loss clamping

    Args:
        cross_entropy_weight (float): Weight for standard cross-entropy loss.
        target_cross_entropy_weight (float): Weight for target-based loss (used in targeted UAP).
        clamp (float or None): If set, clamps individual sample losses to this maximum.
    """
    def __init__(self,
        cross_entropy_weight=1,
        target_cross_entropy_weight=0,
        clamp=12.0,  # float or None
        ):
        super(LossFunction, self).__init__()

        self.cross_entropy_weight = cross_entropy_weight
        self.target_cross_entropy_weight = target_cross_entropy_weight
        self.clamp = clamp


    def forward(self, output, y_label, target,orin_output=None,device=None):
        """
        Compute the final loss value.

        Args:
            output (Tensor): Model output logits. Shape: (B, num_classes)
            y_label (Tensor): Ground-truth class labels. Shape: (B,)
            target (int or None): Target class index for targeted loss.
            orin_output: (Unused)
            device (torch.device or None): Optional device override.

        Returns:
            Tensor: Scalar loss value (mean across batch).
        """
        if device is None:
            device = output.device

        loss = torch.zeros(output.shape[0]).to(device)

        # Apply standard cross-entropy loss
        if self.cross_entropy_weight > 0:
            loss += self.cross_entropy_weight * cross_entropy_fn(output, y_label, target)
            
        # Apply target class loss (negative CE for targeted attacks)
        if self.target_cross_entropy_weight > 0:
            loss += self.target_cross_entropy_weight * target_cross_entropy_fn(output, y_label, target)

        # Clamp individual sample losses if specified
        if self.clamp is not None:
            clamp_tensor = torch.tensor(self.clamp, device=loss.device)
            loss = torch.min(loss, clamp_tensor)

        return torch.mean(loss)
    
    def get_clamp(self):
        return self.clamp


def cross_entropy_fn(output, y_label, target):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    return loss_fn(output, y_label)


def target_cross_entropy_fn(output, y_label, target):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    target_tensor = torch.full((output.shape[0],), target, device=output.device)
    return -loss_fn(output, target_tensor)

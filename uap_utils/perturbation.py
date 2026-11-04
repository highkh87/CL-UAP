import torch


def apply_perturbation(image, delta, clamp_range=(0, 1)):
    """
    Apply adversarial perturbation to an image and clamp the result.

    Args:
        image (Tensor): Original image tensor of shape (B, C, H, W) or (C, H, W).
        delta (Tensor): Perturbation tensor of the same shape as image.
        clamp_range (tuple): Tuple of (min, max) values to clamp output.

    Returns:
        Tensor: Perturbed and clamped image.
    """
    perturbed = image + delta
    perturbed = torch.clamp(perturbed, clamp_range[0], clamp_range[1])
    return perturbed


def visualize_uap(delta, eps=10/255):
    """
    Normalize a universal adversarial perturbation (UAP) to [0, 1] range for visualization.

    Args:
        delta (Tensor): Perturbation tensor of shape (C, H, W), values in [-eps, eps].
        eps (float): Maximum L-infinity bound (used for normalization).

    Returns:
        Tensor: Normalized image for visualization (values in [0, 1]).
    """
    delta_img = delta.detach().cpu().clone()
    
    # Rescale from [-eps, eps] → [0, 1]
    delta_img = ((delta_img / eps) + 1) / 2
    
    return delta_img
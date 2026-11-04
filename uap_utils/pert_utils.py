import torch


def tile_pert_to_image(pert, image_height=224, image_width=224):
    """
    Tiles a smaller pert across a larger image canvas.

    Args:
        pert (Tensor): Input pert tensor of shape (C, pert_H, pert_W).
        image_height (int): Desired output image height (default: 224).
        image_width (int): Desired output image width (default: 224).

    Returns:
        Tensor: Tiled image of shape (C, image_height, image_width).
    """
    C, pert_H, pert_W = pert.shape
    
    # Determine number of repetitions needed in height and width
    n_H = (image_height + pert_H - 1) // pert_H
    n_W = (image_width + pert_W - 1) // pert_W

    # Repeat the pert into a grid of shape (n_H, n_W)
    tiled = pert.unsqueeze(0).repeat(n_H * n_W, 1, 1, 1)
    tiled = tiled.view(n_H, n_W, C, pert_H, pert_W)
    
    # Rearrange and reshape to get full tiled image
    tiled = tiled.permute(2, 0, 3, 1, 4).contiguous().view(C, n_H * pert_H, n_W * pert_W)
    
    # Crop to desired image size
    return tiled[:, :image_height, :image_width]

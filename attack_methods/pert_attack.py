import logging
import torch
import time
from tqdm import tqdm
import numpy as np
from itertools import groupby

from uap_utils.pert_utils import tile_pert_to_image

logger = logging.getLogger("uap_logger")


def growing_pert_attack(
        model,
        loader,
        nb_epoch,
        eps,
        learning_scheduler,
        y_target=None,
        loss_fn=None,
        uap_init = None,
        uap_size=(3, 224, 224),
        start_pert_size=7,
        final_pert_size=224,
        growing_pert_interval=1,
        device='cuda',
        args=None,
        ):
    """
    Generate Universal Adversarial Perturbation (UAP) using growing pert optimization.

    Args:
        model (torch.nn.Module): Target model to attack.
        loader (DataLoader): Dataloader for training images.
        nb_epoch (int): Number of optimization epochs.
        eps (float): Maximum perturbation value (L-infinity norm bound).
        learning_scheduler (list of float): Learning rate schedule per epoch.
        y_target (int or None): Target class for targeted attack. If None, performs untargeted attack.
        loss_fn (callable): Loss function that takes (logits, labels, y_target).
        uap_init (torch.Tensor or None): Optional initial pert to start from.
        uap_size (tuple): Shape of the full perturbation (default is (3, 224, 224)).
        start_pert_size (int): Initial pert size (e.g., 7x7).
        final_pert_size (int): Final pert size (typically 224x224).
        growing_pert_interval (int): Number of epochs between pert size doubling.
        device (str): Device to use (e.g., "cuda" or "cpu").
        args (Namespace): Optional argparse arguments.

    Returns:
        torch.Tensor: Final adversarial perturbation (UAP).
        list of float: Loss value per iteration.
        list of torch.Tensor: UAPs stored after each epoch (useful for visualization).
    """
    model.eval()  # Do not train the model
    
    # Get input shape from a batch
    _, (image, _) = next(enumerate(loader))
    B, C, H, W = image.shape

    # Initialize pert
    pert_size = start_pert_size
    pert_delta = torch.zeros((3, pert_size, pert_size), device=device)

    losses = []
    uaps = []
    
    total_start_time = time.time()
    
    for epoch in range(nb_epoch):
        epoch_start_time = time.time()
        
        learning_rate = learning_scheduler[epoch]
        epoch_losses = []
        
        for i, (x_val, y_val) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{nb_epoch}"):
            x_val, y_val = x_val.to(device), y_val.to(device)
            model.eval()
            
            # Reset gradient
            if pert_delta.grad is not None:
                pert_delta.grad.zero_()
            pert_delta.requires_grad_()
            
            # Expand pert to full image
            delta = tile_pert_to_image(pert_delta)

            # Repeat for batch
            batch_delta = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1]).requires_grad_()

            # Apply perturbation and clamp
            perturbed = torch.clamp(x_val + batch_delta, 0, 1)

            # Forward pass
            outputs = model(perturbed)
            
            # Compute loss
            loss = loss_fn(outputs, y_val, y_target)
            losses.append(loss.item())
            epoch_losses.append(loss.item())
            
            # Backward
            loss.backward()
            
            # Gradient update (sign-based)
            grad = pert_delta.grad.data.sign()
            with torch.no_grad():
                pert_delta += grad * learning_rate
                pert_delta = torch.clamp(pert_delta, -eps, eps)
            
        logger.info(f"epoch {epoch+1}/{nb_epoch}\tloss: {np.mean(epoch_losses)}\tepoch_time: {time.time() - epoch_start_time}")
        
        # Grow pert size if needed
        if (epoch+1) % growing_pert_interval == 0 and pert_size < final_pert_size:
            pert_size *= 2
            pert_delta = tile_pert_to_image(pert_delta, pert_size, pert_size)
        
        # Save intermediate UAP
        tiled_delta = tile_pert_to_image(pert_delta, 224, 224)
        uaps.append(tiled_delta.data.cpu())
    
    logger.info(f"Total training time: {time.time() - total_start_time}")
    
    # Final UAP (tiled to 224x224)
    pert_delta = tile_pert_to_image(pert_delta, 224, 224)
    uap = pert_delta.data.cpu()
    
    return uap, losses, uaps

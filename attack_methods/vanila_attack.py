import logging
import torch
import time
from tqdm import tqdm

logger = logging.getLogger("uap_logger")


def vanila_attack(
        model,
        loader,
        nb_epoch,
        eps,
        learning_scheduler,
        y_target=None,
        loss_fn=None,
        uap_init = None,
        uap_size=(3, 224, 224),
        device='cuda',
        args=None,
        ):
    """
    Generate Universal Adversarial Perturbation (UAP) using simple gradient ascent.

    Args:
        model (torch.nn.Module): Target model to attack.
        loader (DataLoader): Dataloader for training images.
        nb_epoch (int): Number of optimization epochs.
        eps (float): Maximum perturbation magnitude (L-infinity norm bound).
        learning_scheduler (list of float): Learning rate for each epoch.
        y_target (int or None): Target class label (for targeted attacks).
        loss_fn (callable): Loss function to optimize.
        uap_init (torch.Tensor or None): Optional initial perturbation.
        uap_size (int, tuple or None): Desired size of UAP.
        device (str): Device to run optimization on.
        args (Namespace or None): Optional argparse arguments.

    Returns:
        uap (torch.Tensor): Final optimized universal perturbation.
        losses (list of float): Loss value per iteration.
    """
    model.eval()
    
    logger.debug("===================== vanila =====================")
    
    # Infer image shape from one batch
    _, (image, _) = next(enumerate(loader))
    B, C, H, W = image.shape
    
    # Normalize uap_size to (C, H, W) format
    if uap_size is None:
        uap_size = (C, H, W)
    elif isinstance(uap_size, int):
        uap_size = (C, uap_size, uap_size)
    if isinstance(uap_size, (tuple, list)):
        if len(uap_size) == 2:
            uap_size = (C, uap_size[0], uap_size[1])
        elif len(uap_size) == 3:
            uap_size = (uap_size[0], uap_size[1], uap_size[2])
        else:
            raise ValueError(f"pert_size must have 2 or 3 elements, but got {len(uap_size)}.")
    else:
        raise TypeError(f"pert_size must be None, int, tuple, or list, but got {type(uap_size)}.")

    # Initialize perturbation
    if uap_init is None:
        delta = torch.zeros(uap_size, device=device)
    else:
        delta = uap_init.to(device)
        
    losses = []
    total_start_time = time.time()
    
    for epoch in range(nb_epoch):
        epoch_start_time = time.time()
        
        learning_rate = learning_scheduler[epoch]
        for i, (x_val, y_val) in tqdm(enumerate(loader), total=len(loader), desc=f"epoch {epoch+1}/{nb_epoch}"):
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            model.eval()
            
            if delta.grad is not None:
                delta.grad.zero_()
            delta.requires_grad_()

            # Expand to batch dimension
            batch_delta = delta.detach().clone().unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1]).requires_grad_()

            # Apply perturbation and clamp
            perturbed = torch.clamp(x_val + batch_delta, 0, 1)

            # Forward pass
            outputs = model(perturbed)
            
            # Compute loss
            loss = loss_fn(outputs, y_val, y_target)
            losses.append(loss.item())
            
            # Backward pass
            loss.backward()

            # Update using sign of mean gradient
            grad = batch_delta.grad.data.mean(dim=0)
            grad = grad.sign()
            
            with torch.no_grad():
                delta += grad * learning_rate
                delta = torch.clamp(delta, -eps, eps)
            
        logger.info(f"epoch {epoch+1}/{nb_epoch}\tloss: {loss}\tepoch_time: {time.time() - epoch_start_time}")
    
    logger.info(f"Total training time: {time.time() - total_start_time}")
    
    uap = delta.data.cpu()
    
    # Optionally discretize UAP if needed
    # if args.discretize:
    #     uap = torch.round(uap * 255) / 255
    
    return uap, losses

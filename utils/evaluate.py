import numpy as np
import math
from tqdm import tqdm

import torch


def evaluate(
    model, 
    loader, 
    uap = None, 
    K = 5, 
    colorspace=None, 
    eval_original=False, 
    eval_perturbed=True, 
    args=None
    ):
    """
    Evaluate a model on a dataset with and/or without a universal adversarial perturbation (UAP).

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for evaluation.
        uap (Tensor or None): Universal perturbation to apply.
        K (int): Number of top predictions to consider (for top-K accuracy).
        colorspace: (Unused) Placeholder for future use.
        eval_original (bool): Whether to evaluate on clean images.
        eval_perturbed (bool): Whether to evaluate on perturbed images.
        args (Namespace or None): Optional arguments, e.g., pert_learning flag.

    Returns:
        dict: A dictionary containing predictions, top-K results, and accuracy flags.
    """
    original_probs, perturbed_probs, labels = [], [], []
    model.eval()
    
    # Adjust UAP size if using pert learning
    _, (x_val, y_val) = next(enumerate(loader))
    b, c, h, w = x_val.shape
    
    if uap is not None:
        if args and getattr(args, 'pert_learning', False):
            pert_size = args.pert_size
            repeat_h = h // pert_size
            repeat_w = w // pert_size
            uap = uap.repeat(1, repeat_h, repeat_w)  # (C, H, W)
        uap = uap.unsqueeze(0)  # (B, C, H, W)
    else:
        uap = torch.zeros_like(x_val)
    
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in tqdm(enumerate(loader), total=len(loader), desc="evaluate"):
            b, c, h, w = x_val.shape
            
            # Apply UAP if available; else use zero perturbation
            if uap is not None:
                batched_uap = uap.repeat(b, 1, 1, 1)
            else:
                batched_uap = torch.zeros_like(x_val)
            
            labels.append(y_val)

            if eval_original:
                original_out = torch.nn.functional.softmax(model(x_val.cuda()), dim = 1)
                original_probs.append(original_out.cpu().numpy())

            if batched_uap is not None and eval_perturbed:
                perturbed = torch.clamp(x_val + batched_uap, 0, 1)
                perturbed_out = torch.nn.functional.softmax(model(perturbed), dim=1)
                perturbed_probs.append(perturbed_out.cpu().numpy())
            
    # Flatten batches into full numpy arrays 
    original_probs = np.stack([p for l in original_probs for p in l])
    perturbed_probs = np.stack([p for l in perturbed_probs for p in l])
    labels = np.array([t for l in labels for t in l])

    res = {
        'original_probs': original_probs,
        'perturbed_probs': perturbed_probs,
        'labels': labels,
        'K': K
    }
    
    if eval_original:
        res['original_top'] = original_top = np.argpartition(-original_probs, K, axis = 1)[:,:K]
        res['original_top_probs'] = original_top_probs = original_probs[np.arange(original_probs.shape[0])[:, None], original_top].astype(np.float16)
        res['original_top1acc'] = original_top[range(len(original_top)), np.argmax(original_top_probs, axis = 1)] == labels
        res['original_topKacc'] = [labels[i] in row for i, row in enumerate(original_top)]
        res['original_outputs'] = original_top[range(len(original_top)), np.argmax(original_top_probs, axis = 1)]
        
    if uap is not None and eval_perturbed:
        res['perturbed_top'] = perturbed_top = np.argpartition(-perturbed_probs, K, axis = 1)[:,:K]
        res['perturbed_top_probs'] = perturbed_top_probs = perturbed_probs[np.arange(perturbed_probs.shape[0])[:, None], perturbed_top].astype(np.float16)
        res['perturbed_top1acc'] = perturbed_top[range(len(perturbed_top)), np.argmax(perturbed_top_probs, axis = 1)] == labels
        res['perturbed_topKacc'] = [labels[i] in row for i, row in enumerate(perturbed_top)]
        res['perturbed_outputs'] = perturbed_top[range(len(perturbed_top)), np.argmax(perturbed_top_probs, axis = 1)]
        
    return res

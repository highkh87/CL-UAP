# coding: utf-8
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models.load_model import load_model
from data_loader.imagenet_loader import imagenet_loader


# ====================================
# Deletion AUC
# ====================================

def simple_deletion_noise(model, image, noise, step=100, image_seed=0):
    model.eval()
    device = image.device
    if noise.dim() == 3:
        noise = noise.unsqueeze(0)
    noise = noise.to(device)
    _, _, H, W = image.shape
    HW = H * W

    sorted_idx = np.random.RandomState(image_seed).permutation(HW)

    deleted_noise = noise.clone().detach()
    scores_target, scores_orig = [], []
    n_steps = (HW + step - 1) // step
    mask = torch.ones((1, 1, H, W), dtype=torch.float32, device=device)

    with torch.no_grad():
        target_class = torch.argmax(model(torch.clamp(image + noise, 0, 1))).item()
        orig_class   = torch.argmax(model(image)).item()
        if target_class == orig_class:
            return None, None
        for i in range(n_steps + 1):
            masked_noise = deleted_noise * mask
            output = model(torch.clamp(image + masked_noise, 0, 1))
            prob_target = F.softmax(output, dim=1)[0, target_class].item()
            prob_orig   = F.softmax(output, dim=1)[0, orig_class].item()
            scores_target.append(prob_target)
            scores_orig.append(prob_orig)
            if i < n_steps:
                idx = sorted_idx[i * step: (i + 1) * step]
                mask_flat = mask.view(1, 1, -1)
                mask_flat[0, 0, idx] = 0
                mask = mask_flat.view(1, 1, H, W)

    return scores_target, scores_orig


# ====================================
# Misc helpers
# ====================================

def cal_auc(p_prob):
    return np.trapz(p_prob, np.arange(len(p_prob))) / len(p_prob)


def load_uap(uap_path, device):
    ext = os.path.splitext(uap_path)[1].lower()
    if ext == ".npy":
        uap = torch.from_numpy(np.load(uap_path))
    elif ext in [".pth", ".pt"]:
        obj = torch.load(uap_path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            uap = obj
        elif isinstance(obj, np.ndarray):
            uap = torch.from_numpy(obj)
        elif isinstance(obj, dict):
            uap = None
            for key in ["uap", "noise", "perturbation"]:
                if key in obj:
                    value = obj[key]
                    uap = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    if uap is not None:
                        break
            if uap is None:
                raise ValueError(f"Cannot find UAP tensor in dict: {uap_path}")
        else:
            raise ValueError(f"Unsupported object type in {uap_path}: {type(obj)}")
    else:
        raise ValueError(f"Unsupported UAP extension: {ext} (use .npy/.pth/.pt)")
    uap = uap.float()
    if uap.dim() not in [3, 4]:
        raise ValueError(f"UAP tensor must be 3D or 4D, got shape={tuple(uap.shape)}")
    return uap.to(device)


def extract_model_name(uap_path):
    known_models = [
        'vit-b-16', 'vit-b-32', 'vit-s-16',
        'deit-b', 'deit-s',
        'resnet152', 'googlenet', 'alexnet', 'vgg16', 'vgg19',
    ]
    for name in known_models:
        if name in uap_path:
            return name
    raise ValueError(f"Cannot determine model name from path: {uap_path}")


# ====================================
# Main
# ====================================

def main():
    parser = argparse.ArgumentParser(
        description="Deletion AUC computation for UAP evaluation (random pixel order)."
    )
    parser.add_argument('--uap_path', type=str, required=True,
                        help='Path to UAP file (.npy/.pth/.pt)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (e.g. vgg16, resnet152). '
                             'Auto-detected from uap_path if not specified.')
    parser.add_argument('--indices_path', type=str, default='./dataset_1000.npy',
                        help='Path to numpy indices file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save AUC logs (default: uap file directory)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for pixel ordering (default: 42)')
    args = parser.parse_args()

    uap_path   = os.path.abspath(args.uap_path)
    model_name = args.model_name if args.model_name else extract_model_name(uap_path)
    log_path   = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(uap_path)
    os.makedirs(log_path, exist_ok=True)
    device = args.device

    val_dir        = '../data/imagenet1K/val'
    val_num_images = 50000
    nums       = np.load(args.indices_path)
    val_loader = imagenet_loader(val_dir, val_num_images, 1, shuffle=False)

    print(f"==> UAP: {uap_path}")
    print(f"==> Model: {model_name} | Seed: {args.seed}")

    model = load_model(model_name).to(device)
    model.eval()
    uap = load_uap(uap_path, device)

    AUC = []
    log_file_path = os.path.join(log_path, f'{model_name}_deletion_auc_random.txt')
    with open(log_file_path, 'w') as f:
        f.write(f"# UAP path: {uap_path}\n")
        for num in tqdm(nums, desc='deletion/random'):
            images, _ = val_loader.dataset[num]
            input_tensor = images.unsqueeze(0).to(device)
            image_seed = args.seed + int(num)
            p_prob, _ = simple_deletion_noise(
                model, input_tensor, uap, step=100, image_seed=image_seed
            )
            auc_score = 0.0 if p_prob is None else cal_auc(p_prob)
            AUC.append(auc_score)
            f.write(f"{num}\t{auc_score:.6f}\n")
        f.write(f"\n# Mean AUC: {np.mean(AUC):.6f}\n")

    print(f"[{model_name}] Deletion  Mean AUC (random): {np.mean(AUC):.6f}  → {log_file_path}")


if __name__ == "__main__":
    main()

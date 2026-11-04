import multiprocessing
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def imagenet_loader(
    dir_data, 
    nb_images = 50000, 
    batch_size = 100, 
    img_size = 224, 
    balanced=True, 
    random_seed=9999, 
    deterministic=True,
    shuffle=True,
    ):
    """
    Returns a DataLoader for ImageNet-compatible directory using optional sampling and preprocessing.

    Args:
        dir_data (str): Path to ImageNet directory (should follow ImageFolder structure).
        nb_images (int): Number of total samples to load (default 50000).
        batch_size (int): Batch size for DataLoader.
        img_size (int): Final image size after center crop.
        balanced (bool): Whether to sample data with class balance.
        random_seed (int): Random seed for reproducibility.
        deterministic (bool): If True, sets deterministic CUDA behavior.
        shuffle (bool): Whether to shuffle data in the DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define preprocessing transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    
    val_dataset = ImageFolder(dir_data, val_transform)
    total_samples = len(val_dataset)

    if nb_images < total_samples:
        np.random.seed(0)

        if balanced:
            # Group sample indices by class
            class_to_indices = defaultdict(list)
            for idx, (_, class_idx) in enumerate(val_dataset.samples):
                class_to_indices[class_idx].append(idx)
            
            num_classes = len(class_to_indices)
            samples_per_class = nb_images // num_classes
            selected_indices = []
            class_used_counts = {} # Track how many samples used class

            # Sample evenly from each class
            for class_idx, indices in class_to_indices.items():
                np.random.shuffle(indices)
                n_to_sample = min(samples_per_class, len(indices))
                selected_indices.extend(indices[:n_to_sample])
                class_used_counts[class_idx] = n_to_sample
            
            # Fill remaining quota with one sample per class
            remaining = nb_images - len(selected_indices)
            class_indices = list(class_to_indices.keys())
            np.random.shuffle(class_indices)
            i = 0
            while remaining > 0:
                class_idx = class_indices[i]
                if class_used_counts[class_idx] == len(class_to_indices[class_idx]):
                    i = (i + 1) % len(class_indices)
                    continue
                
                n_to_sample = class_used_counts[class_idx]
                selected_indices.append(class_to_indices[class_idx][n_to_sample])
                class_used_counts[class_idx] += 1
                remaining -= 1
                i = (i + 1) % len(class_indices)
            
        else:
            # Random sampling without balancing
            selected_indices = np.random.permutation(range(total_samples))[:nb_images]
            
        val_dataset = Subset(val_dataset, selected_indices)
        
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = shuffle, 
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    )
    
    return dataloader
import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils import model_zoo
import logging

logger = logging.getLogger("uap_logger")
logger.setLevel(logging.DEBUG)

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None].to(tensor.device)
    std = std[None, :, None, None].to(tensor.device)
    return tensor.sub(mean).div(std)


def load_model(model_name, data='imagenet', attach_normalization=True):
    # Transformer based
    if model_name in timm.list_models(pretrained=True):
        model = timm.create_model(model_name, pretrained=True)
        model = nn.DataParallel(model).cuda()
        
    # Load pre-trained ImageNet models from torchvision
    elif model_name=='densenet':
        model = eval("torchvision.models.densenet121(pretrained=True)")
        model = nn.DataParallel(model).cuda()
    elif model_name=='mobilenet':
        model = eval("torchvision.models.mobilenet_v2(pretrained=True)")
        model = nn.DataParallel(model).cuda()
    else:
        model = eval("torchvision.models.{}(pretrained=True)".format(model_name))
        model = nn.DataParallel(model).cuda()
    
    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    if attach_normalization:
        if data == 'imagenet':
            normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)
            model = nn.Sequential(normalize, model)
        if data == 'cifar':
            normalize = Normalizer(mean = CIFAR_MEAN, std = CIFAR_STD)
            model = nn.Sequential(normalize, model)

    model = model.cuda()
    model.model_name = model_name

    logger.debug("Model loading complete.")

    return model

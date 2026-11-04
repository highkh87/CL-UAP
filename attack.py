import argparse
import logging
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.uap_analyzer import UAPAnalyzer
from uap_utils.perturbation import visualize_uap
from utils.evaluate import evaluate

from utils.argparse_utils import str2bool, int_or_none, float_or_none
from data_loader.imagenet_loader import imagenet_loader
from models.load_model import load_model
from losses.loss_function import LossFunction
from attack_methods.vanila_attack import vanila_attack
from attack_methods.pert_attack import growing_pert_attack

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("uap_logger")
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()

    # logging options
    parser.add_argument("--log_dir", type=str, default="./logs/test_log")

    # dataloader options
    parser.add_argument("--train_dir", type=str, default="./data/imagenet1K/train")
    parser.add_argument("--val_dir", type=str, default="./data/imagenet1K/val")
    parser.add_argument("--train_num_images", type=int, default=10000)
    parser.add_argument("--val_num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--balanced", type=str2bool, default=True)
    parser.add_argument("--random_seed",type=int,default=0)
    
    # models
    parser.add_argument("--train_models", type=str, nargs='+', default=["resnet152"])
    parser.add_argument("--val_models", type=str, nargs='+', default=["resnet152"])

    # training parameters
    parser.add_argument("--nb_epoch", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=7)

    # UAP parameters
    parser.add_argument("--eps", type=float, default=10/255)
    parser.add_argument("--pert_size", type=int, default=224)
    parser.add_argument("--start_pert_size", type=int, default=56)
    parser.add_argument("--final_pert_size", type=int, default=224)
    parser.add_argument("--growing_pert_interval", type=int, default=1)

    # loss function optons
    parser.add_argument("--loss_clamp", type=float_or_none, default=12.0)
    parser.add_argument("--target_label", type=int_or_none, default=None)
    parser.add_argument("--cross_entropy_loss_weight", type=float, default=1)
    parser.add_argument("--target_cross_entropy_loss_weight", type=float, default=0)

    # attack method
    parser.add_argument("--attack_method", type=str, default="vanila")
    # options: vanila / growing_pert

    # analysis options
    parser.add_argument("--save_uap", type=str2bool, default=True)
    parser.add_argument("--save_uap_image", type=str2bool, default=True)
    parser.add_argument("--save_loss_image", type=str2bool, default=True)
    # parser.add_argument("--evaluate_uap", type=str2bool, default=True)
    parser.add_argument("--analyze_uap", type=str2bool, default=True)

    args = parser.parse_args()

    return args


def set_logger(args):
    logger_path = f"{args.log_dir}/logfile.log"
    os.makedirs(args.log_dir, exist_ok=True)

    # define logging format
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

    # console handler (DEBUG level and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # file handler (INFO level and above)
    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # example logs
    # logger.debug("This is a DEBUG message")      # only console
    # logger.info("This is an INFO message")       # both console and file
    # logger.warning("This is a WARNING message")  # both
    # logger.error("This is an ERROR message")     # both


def main(args):
    # load data
    train_loader = imagenet_loader(args.train_dir, args.train_num_images, args.batch_size, balanced=args.balanced, random_seed=args.random_seed)
    val_loader = imagenet_loader(args.val_dir, args.val_num_images, args.batch_size, balanced=args.balanced, random_seed=args.random_seed)

    # load models
    train_models, val_models = [], []
    for model_name in args.train_models:
        model = load_model(model_name)
        model.eval()
        train_models.append(model)
    for model_name in args.val_models:
        model = load_model(model_name)
        model.eval()
        val_models.append(model)

    # define loss function
    loss_fn = LossFunction(
        cross_entropy_weight=args.cross_entropy_loss_weight,
        target_cross_entropy_weight=args.target_cross_entropy_loss_weight,
        clamp=args.loss_clamp,
    )
    
    # define learning rate scheduler
    learning_scheduler = [(10/255)*((args.alpha/10) ** i) for i in range(args.nb_epoch)]

    logger.debug(f"eps: {args.eps}")

    # generate UAP
    if args.attack_method == 'vanila':
        uap, losses = vanila_attack(
            train_models[0], 
            train_loader, 
            args.nb_epoch, 
            args.eps, 
            learning_scheduler=learning_scheduler, 
            y_target=args.target_label,
            loss_fn=loss_fn, 
        )
    if args.attack_method == 'growing_pert':
        uap, losses, uaps = growing_pert_attack(
            train_models[0], 
            train_loader, 
            args.nb_epoch, 
            args.eps, 
            learning_scheduler=learning_scheduler, 
            y_target=args.target_label,
            loss_fn=loss_fn,
            start_pert_size=args.start_pert_size,
            final_pert_size=args.final_pert_size,
            growing_pert_interval=args.growing_pert_interval,
        )

    # save UAP tensor
    if args.save_uap:
        torch.save(uap, f"{args.log_dir}/uap.pth")
        if args.attack_method == 'growing_pert':
            for i, _ in enumerate(uaps):
                dir = f"{args.log_dir}/uaps"
                os.makedirs(dir, exist_ok=True)
                torch.save(_, f"{dir}/uap_{i}.pth")

    # save UAP visualization
    if args.save_uap_image:
        uap_image = np.transpose(visualize_uap(uap, args.eps), (1, 2, 0))
        plt.imshow(uap_image)
        plt.axis("off")
        plt.title("Universal Adversarial Perturbation")
        plt.savefig(f"{args.log_dir}/uap_visualization.png", bbox_inches="tight", pad_inches=0.1)
        plt.close()

    # save loss curve
    if args.save_loss_image:
        plt.plot(losses, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("UAP Optimization Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.log_dir}/uap_loss_curve.png", bbox_inches="tight")
        plt.close()
    
    # Example only: manually load and analyze an existing UAP (commented)
    # uap = torch.load("./logs/example/uap.pth")
    
    # analyze UAP using UAPAnalyzer
    if args.analyze_uap:
        analyzer = UAPAnalyzer()
        for model in val_models:
            logger.info(f"{model.model_name} analysis result =======")
            evaluate_result = evaluate(model, val_loader, uap=uap, eval_original=True, eval_perturbed=True)
            labels = evaluate_result['labels']
            original_outputs = evaluate_result['original_outputs']
            perturbed_outputs = evaluate_result['perturbed_outputs']
            result = analyzer.uap_analysis(labels, original_outputs, perturbed_outputs, target_label=args.target_label, print_result=True, save_result=False, show_figure=False, save_figure=True, save_dir=f"{args.log_dir}/analysis_results", save_name=model.model_name)


if __name__ == "__main__":
    args = get_args()

    set_logger(args)

    main(args)

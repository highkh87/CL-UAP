# CL-UAP: Curriculum Learning for Progressive Universal Adversarial Perturbations

This repository contains the official implementation of the WACV 2027 paper:
**CL-UAP: Curriculum Learning for Progressive Universal Adversarial Perturbations**

---

## 📂 Project Structure

```text
CL-UAP/
├── attack_methods/         # CL-UAP and vanilla UAP generation logic
├── data_loader/            # ImageNet dataloader with class-balanced sampling
├── losses/                 # Loss functions for UAP optimization
├── models/                 # Target model loading and input normalization
├── scripts/                # Shell script for UAP generation
├── pid_auc/                # Python scripts for insertion and deletion AUC
├── uap_utils/              # Perturbation tiling and visualization utilities
├── utils/                  # Argument parsing, evaluation, and analysis utilities
├── attack.py               # Main script for UAP generation and evaluation
├── requirements.txt
├── data/                   # Dataset directory to be prepared separately
├── LICENSE
└── README.md
```

The ImageNet dataset is not included in this repository.

---

## 🚀 Getting Started

### 1. Installation

The current implementation requires a CUDA-capable GPU and a compatible PyTorch installation.

```bash
git clone https://github.com/highkh87/CL-UAP.git
cd CL-UAP
pip install -r requirements.txt
```

### 2. Dataset Preparation

Prepare the following files and folders:

```text
./data/imagenet1K/train/           # Training images for UAP generation
./data/imagenet1K/val/             # Validation images for evaluation
./data/imagenet_class_index.json   # ImageNet class index for analysis
```

The training and validation directories must follow the `torchvision.datasets.ImageFolder` structure, with images organized into class subdirectories. The validation set must use the same class-to-index mapping as the training set.

You may create symbolic links if your data is stored elsewhere. Custom dataset paths can also be specified using `--train_dir` and `--val_dir` when running `attack.py`.

---

## 🛠️ Usage

Run the following commands from the repository root unless otherwise specified.

### 1. Generate UAPs via Shell Script

```bash
bash scripts/run_attack_CL-UAP.sh
```

This script launches `attack.py` with predefined hyperparameters for CL-UAP generation. Adjust the GPU device, model, and other settings in the script as needed.

### 2. Generate UAPs Manually

```bash
python attack.py \
    --attack_method growing_pert \
    --start_pert_size 28 \
    --final_pert_size 224 \
    --nb_epoch 20 \
    --batch_size 64 \
    --alpha 7 \
    --train_models resnet152 \
    --val_models resnet152 \
    --log_dir ./logs/my_run
```

By default, `attack.py` saves the generated UAP and visualizations, then evaluates and analyzes the UAP on the specified validation models.

---

## 🧪 Evaluation

Evaluation and analysis are implemented in two modules:

- `utils/evaluate.py`: evaluates clean and perturbed images and returns predictions, probabilities, and per-sample top-1/top-K correctness indicators.
- `utils/uap_analyzer.py`: computes accuracy, fooling rate, and additional statistics from the evaluation outputs.

By default, `attack.py` calls these modules after UAP generation. The current `attack.py` does not provide an evaluation-only command-line option. To evaluate an existing UAP, load it in your own Python script and call the evaluation and analysis utilities.

---

## 🧠 Additional Notes

- Perturbation tiling and visualization utilities are in `uap_utils/`.
- Argument parsing, evaluation, and analysis utilities are in `utils/`.
- Loss functions used during optimization are in `losses/`.
- Target models are loaded through `models/load_model.py` using torchvision or timm.

---

## 📓 PID AUC (Optional)

Insertion and deletion AUC can be computed using the Python scripts in `pid_auc/`. These analyses are optional and are not required for core UAP generation or evaluation. Please refer to the paper for the detailed evaluation protocol.

Run the following commands from the `pid_auc/` directory:

```bash
cd pid_auc

python insertion_auc.py \
    --uap_path ../logs/my_run/uap.pth \
    --model_name resnet152

python deletion_auc.py \
    --uap_path ../logs/my_run/uap.pth \
    --model_name resnet152
```

These examples assume the default validation dataset location and use the included `dataset_1000.npy` index file. Replace the UAP path and model name as needed.

---

## 📜 License

This repository is released under the [MIT License](LICENSE).

from torchvision.datasets.utils import download_url
import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from collections import Counter
import os
import torch
from scipy.stats import entropy
import argparse
import logging

logger = logging.getLogger("uap_logger")


class UAPAnalyzer():
    def __init__(self, dataset='ImageNet', imagenet_class_index_path='./data/imagenet_class_index.json'):

        if dataset == 'ImageNet':
            # Download and load ImageNet class index if not available
            with open(imagenet_class_index_path) as f:
                class_idx = json.load(f)
            
            nltk.download('wordnet')
            
            # Convert class index to WordNet synsets
            self.imagenet_synsets = []
            for idx in range(1000):
                wnid = class_idx[str(idx)][0]
                offset = int(wnid[1:])
                synset = wn.synset_from_pos_and_offset('n', offset)
                self.imagenet_synsets.append(synset)
            
            self.num_classes = len(self.imagenet_synsets)
        
        else:
            pass


    def uap_analysis(
            self, labels, outputs, attack_outputs,
            save_dir='./results', save_name='result',
            K=5, target_label=None,
            calculate_clip_similarity=False,
            print_result=True, save_result=False, show_figure=True, W=20, H=12, save_figure=False):
        
        """
        Perform detailed UAP performance analysis and generate summary + visualizations.
        """
        save_name = save_name.replace("/", "_")
        if save_result or save_figure:
            os.makedirs(save_dir, exist_ok=True)

        # Compute accuracy and fooling metrics
        accuracy = sum(labels == outputs) / len(labels)
        attack_accuracy = sum(labels == attack_outputs) / len(labels)
        absolute_drop = accuracy - attack_accuracy
        relative_drop = absolute_drop / accuracy
        fooling_rate = sum(outputs != attack_outputs) / len(outputs)
        target_accuracy = sum(attack_outputs == target_label) / len(labels) if target_label else None

        results = {
            'accuracy': accuracy,
            'attack_accuracy': attack_accuracy,
            'absolute_drop': absolute_drop,
            'relative_drop': relative_drop,
            'fooling_rate': fooling_rate,
            'target_accuracy': target_accuracy,
        }

        # Count label distributions
        label_counts = Counter({i: 0 for i in range(self.num_classes)})
        label_counts.update(labels)
        output_counts = Counter({i: 0 for i in range(self.num_classes)})
        output_counts.update(outputs)
        attack_output_counts = Counter({i: 0 for i in range(self.num_classes)})
        attack_output_counts.update(attack_outputs)

        sorted_attack_outputs = sorted(attack_output_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_attack_outputs = sorted_attack_outputs[:K]
        top_k_ratio = [(label, count / len(attack_outputs)) for label, count in top_k_attack_outputs]

        # Analyze high-class changes after attack
        attack_counter = Counter(attack_outputs)
        total_high_class, total_high_class_count = attack_counter.most_common(1)[0]
        total_high_class_ratio = total_high_class_count / len(attack_outputs)
        
        fooling_outputs = [a for o, a in zip(outputs, attack_outputs) if o != a]
        fooling_counter = Counter(fooling_outputs)
        if fooling_counter:
            fooling_high_class, fooling_high_class_count = fooling_counter.most_common(1)[0]
            fooling_high_class_ratio = fooling_high_class_count / len(fooling_outputs)
        else:
            fooling_high_class, fooling_high_class_ratio = None, 0

        # Print or save results
        if print_result or save_result:
            result_str = ""
            result_str += '======================= Analysis Results =======================\n'
            result_str += f"Original Accuracy: \t{accuracy:}\n"
            result_str += f"Attack Accuracy: \t{attack_accuracy}\n"
            result_str += f"Absolute Drop: \t{absolute_drop}\n"
            result_str += f"Relative Drop: \t{relative_drop}\n"
            result_str += f"Fooling Rate: \t{fooling_rate}\n"
            result_str += f"Fooling high-class: {fooling_high_class}({self.imagenet_synsets[fooling_high_class].lemmas()[0].name()})\n"
            result_str += f"Fooling high-class ratio: {fooling_high_class_ratio}\n"
            result_str += f"Total high-class: {total_high_class}({self.imagenet_synsets[total_high_class].lemmas()[0].name()})\n"
            result_str += f"Total high-class ratio: {total_high_class_ratio}\n"
            if target_label: result_str += f"Target Accuracy: \t{target_accuracy}\n"
            if calculate_clip_similarity:
                result_str += f"Original CLIP similarity's mean: \t{results['outputs_similarity_mean']}\n"
                result_str += f"Original CLIP similarity's std: \t{results['outputs_similarity_std']}\n"
                result_str += f"Attack CLIP similarity's mean: \t{results['attack_outputs_similarity_mean']}\n"
                result_str += f"Attack CLIP similarity's std: \t{results['attack_outputs_similarity_std']}\n"
            result_str += '\n'

            if print_result: logger.info(result_str)
            if save_result:
                with open(os.path.join(save_dir, f"{save_name}_analysis_results.txt"), "w") as f:
                    f.write(result_str)

        # Visualize histograms and label distributions
        if show_figure or save_figure:
            interval = self.num_classes / 10
            
            class_ids, class_counts = zip(*label_counts.items())
            plt.figure(figsize=(W, H))
            plt.bar(class_ids, class_counts)
            plt.xlabel("Predicted Labels")
            plt.ylabel("Count")
            plt.title("Labels Distribution")
            plt.xlim(0, 1000)
            plt.xticks(ticks=np.arange(0, len(class_ids), interval), labels=np.arange(0, len(class_ids), interval), rotation=45)
            if save_figure: plt.savefig(os.path.join(save_dir, f"{save_name}_Labels Distribution.png"), dpi=300)
            plt.show()

            # Outputs Distribution
            class_ids, class_counts = zip(*output_counts.items())
            plt.figure(figsize=(W, H))
            plt.bar(class_ids, class_counts)
            plt.xlabel("Predicted Labels")
            plt.ylabel("Count")
            plt.title("Outputs Distribution")
            plt.xlim(0, 1000)
            plt.xticks(ticks=np.arange(0, len(class_ids), interval), labels=np.arange(0, len(class_ids), interval), rotation=45)
            if save_figure: plt.savefig(os.path.join(save_dir, f"{save_name}_Outputs Distribution.png"), dpi=300)
            plt.show()

            # Attack_outputs Distribution
            class_ids, class_counts = zip(*attack_output_counts.items())
            plt.figure(figsize=(W, H))
            plt.bar(class_ids, class_counts)
            plt.xlabel("Predicted Labels")
            plt.ylabel("Count")
            plt.title("Attack_outputs Distribution")
            plt.xlim(0, 1000)
            plt.xticks(ticks=np.arange(0, len(class_ids), interval), labels=np.arange(0, len(class_ids), interval), rotation=45)
            if save_figure: plt.savefig(os.path.join(save_dir, f"{save_name}_Attack_outputs Distribution.png"), dpi=300)
            plt.show()

            # Top-K class histogram
            plt.figure(figsize=(W, H))
            sns.barplot(x=[x[0] for x in top_k_ratio], y=[x[1] for x in top_k_ratio])
            plt.xlabel("Label")
            plt.ylabel("Percentage (%)")
            plt.title(f"Top-{K} Label Distribution")
            plt.xticks(rotation=45)
            if save_figure: plt.savefig(os.path.join(save_dir, f"{save_name}_Top-{K} Label Distribution.png"), dpi=300)
            plt.show()

        return results

import argparse
import os
import pprint

import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt

from bridge_vars import NUM_CALLS
from nets import PolicyNet
import common_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str,
                        default=r"D:/RL/rlul/pyrlul/bridge/dataset/expert/sl_data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="imitation_learning/metrics")
    return parser.parse_args()


def test_and_compute_metrics():
    args = parse_args()
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    obs = torch.load(os.path.join(args.dataset_dir, "test_obs.p"))["s"]
    labels = torch.load(os.path.join(args.dataset_dir, "test_label.p"))

    net = PolicyNet()
    net.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"])
    net.to(args.device)
    with torch.no_grad():
        log_probs = net(obs.to(args.device))
    log_probs = log_probs.cpu()
    print(log_probs)
    probs = torch.exp(log_probs)
    pred_actions = torch.argmax(probs, 1)
    accuracy = (pred_actions == labels).int().sum() / pred_actions.shape[0]

    confusion_matrix = torchmetrics.functional.classification.multiclass_confusion_matrix(pred_actions,
                                                                                          labels, NUM_CALLS, "none")

    # plot_confusion_matrix(confusion_matrix.numpy(), classes_str, True)

    stats = torchmetrics.functional.classification.multiclass_stat_scores(pred_actions, labels, NUM_CALLS, "none")

    # Compute TP, FP, TN, and FN for each class
    tp = stats[:, 0].squeeze()
    fp = stats[:, 1].squeeze()
    tn = stats[:, 2].squeeze()
    fn = stats[:, 3].squeeze()
    label_counts = stats[:, 4].squeeze()

    # p, r, f1 for each class

    precision = torchmetrics.functional.classification.multiclass_precision(pred_actions, labels, NUM_CALLS, "none")
    recall = torchmetrics.functional.classification.multiclass_recall(pred_actions, labels, NUM_CALLS, "none")
    f1_score = torchmetrics.functional.classification.multiclass_f1_score(pred_actions, labels, NUM_CALLS, "none")

    # plt.figure(figsize=(50, 50))
    plt.plot(np.arange(0, NUM_CALLS), precision)
    plt.xlabel("ordered bids")
    plt.ylabel("precision")
    plt.title("Precision")
    plt.ylim(0, 1)
    # plt.xticks(tick_marks, classes_str, rotation=45)
    plt.savefig(os.path.join(save_dir, "precision.png"))
    plt.close()
    plt.figure()
    plt.plot(np.arange(0, NUM_CALLS), recall)
    plt.xlabel("ordered bids")
    plt.ylabel("recall")
    plt.ylim(0, 1)
    plt.title("Recall")
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.show()
    plt.savefig(os.path.join(save_dir, "recall.png"))
    plt.close()
    plt.figure()
    plt.plot(np.arange(0, NUM_CALLS), f1_score)
    plt.xlabel("ordered bids")
    plt.ylabel("f1-score")
    plt.ylim(0, 1)
    plt.title("F1-score")
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.show()
    plt.savefig(os.path.join(save_dir, "f1-score.png"))
    plt.close()

    # marco p, r, f1
    macro_averaged_precision = torchmetrics.functional.classification.multiclass_precision(pred_actions, labels,
                                                                                           NUM_CALLS, "macro")
    macro_averaged_recall = torchmetrics.functional.classification.multiclass_recall(pred_actions, labels,
                                                                                     NUM_CALLS, "macro")
    macro_averaged_f1 = torchmetrics.functional.classification.multiclass_f1_score(pred_actions, labels,
                                                                                   NUM_CALLS, "macro")

    # micro p, r, f1
    micro_averaged_precision = torchmetrics.functional.classification.multiclass_precision(pred_actions, labels,
                                                                                           NUM_CALLS, "micro")

    micro_averaged_recall = torchmetrics.functional.classification.multiclass_recall(pred_actions, labels,
                                                                                     NUM_CALLS, "micro")
    micro_averaged_f1 = torchmetrics.functional.classification.multiclass_f1_score(pred_actions, labels,
                                                                                   NUM_CALLS, "micro")

    # auc for each class
    auc_per_class = torchmetrics.functional.classification.multiclass_auroc(probs, labels, NUM_CALLS, "none")
    plt.figure()
    plt.plot(np.arange(0, NUM_CALLS), auc_per_class)
    plt.xlabel("ordered bids")
    plt.ylabel("auc")
    plt.title("AUC")
    plt.ylim(0.99, 1)
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.show()
    plt.savefig(os.path.join(save_dir, "auc.png"))
    plt.close()

    stats = {
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "label_counts": label_counts,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "micro_precision": micro_averaged_precision,
        "micro_recall": micro_averaged_recall,
        "micro_f1": micro_averaged_f1,
        "macro_precision": macro_averaged_precision,
        "macro_recall": macro_averaged_recall,
        "macro_f1": macro_averaged_f1,
        "auc": auc_per_class,
        "confusion_matrix": confusion_matrix
    }
    torch.save(stats, os.path.join(save_dir, "stats.pth"))
    pprint.pprint(stats)

if __name__ == '__main__':
    test_and_compute_metrics()
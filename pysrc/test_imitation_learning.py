import argparse
import os
import pprint

import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt

from bridge_consts import NUM_CALLS
from nets import PolicyNet, PolicyNet2
import common_utils
from utils import sl_net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir", type=str, default="imitation_learning/folder_5"
    )
    parser.add_argument("--dataset_dir", type=str, default=r"expert/sl_data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="imitation_learning/metrics")
    return parser.parse_args()


def test():
    args = parse_args()
    model_paths = []
    for root, dirs, files in os.walk(args.checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith("pth"):
                model_paths.append(file_path)
    torch.set_printoptions(threshold=10000000)
    print(model_paths)

    dataset = torch.load(os.path.join(args.dataset_dir, "test_obs.p"))
    # obs = dataset["s"]
    obs = dataset["s"]
    labels = torch.load(os.path.join(args.dataset_dir, "test_label.p"))
    net = PolicyNet()
    net.to(args.device)
    for model_path in model_paths:
        net.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            log_probs = net(obs.to(args.device))
            log_probs = log_probs.cpu()
            probs = torch.exp(log_probs)
            pred_actions = torch.argmax(probs, 1)
            accuracy = (pred_actions == labels).int().sum() / pred_actions.shape[0]
            # get_metrics(probs, labels)
        print(f"{model_path}: {accuracy}")
        # net = sl_net()
        #
        # log_probs = log_probs.cpu()
        # # print(log_probs)
        # probs = torch.exp(log_probs)
        # # print(probs)
        # pred_actions = torch.argmax(probs, 1)
        # accuracy = (pred_actions == labels).int().sum() / pred_actions.shape[0]
        #
        eq_indices = torch.nonzero(pred_actions == labels).squeeze()
        neq_indices = torch.nonzero(pred_actions != labels).squeeze()
        correct_classification_label_probs = (
            probs[eq_indices].gather(1, labels[eq_indices].unsqueeze(1)).squeeze(1)
        )
        print(
            f"correct_classification_label_probs:"
            f"max: {correct_classification_label_probs.max()},"
            f"min : {correct_classification_label_probs.min()},"
            f"mean : {correct_classification_label_probs.mean()},"
            f"std: {torch.std(correct_classification_label_probs)}"
        )
        incorrect_classification_label_probs = (
            probs[neq_indices].gather(1, labels[neq_indices].unsqueeze(1)).squeeze(1)
        )
        print(
            f"incorrect_classification_label_probs:"
            f"max: {incorrect_classification_label_probs.max()},"
            f"min : {incorrect_classification_label_probs.min()},"
            f"mean : {incorrect_classification_label_probs.mean()},"
            f"std: {torch.std(incorrect_classification_label_probs)}"
        )

        print(neq_indices.shape)
        # print(labels[neq_indices])


def test_and_compute_metrics():
    torch.set_printoptions(threshold=10000000)
    args = parse_args()
    save_dir = common_utils.mkdir_with_increment("figs")
    dataset = torch.load(os.path.join(args.dataset_dir, "test_obs.p"))
    obs = dataset["s"]
    labels = torch.load(os.path.join(args.dataset_dir, "test_label.p"))

    net = PolicyNet2()
    net.load_state_dict(torch.load("models/il_net.pth"))
    net.to(args.device)
    with torch.no_grad():
        log_probs = net(obs.to(args.device))
    log_probs = log_probs.cpu()
    # print(log_probs)
    probs = torch.exp(log_probs)
    # print(probs)
    get_metrics(probs, labels, save_dir)


def get_metrics(probs: torch.Tensor, labels: torch.Tensor, save_dir):
    pred_actions = torch.argmax(probs, 1)
    accuracy = (pred_actions == labels).int().sum() / pred_actions.shape[0]

    confusion_matrix = (
        torchmetrics.functional.classification.multiclass_confusion_matrix(
            pred_actions, labels, NUM_CALLS, "none"
        )
    )

    # plot_confusion_matrix(confusion_matrix.numpy(), classes_str, True)

    stats = torchmetrics.functional.classification.multiclass_stat_scores(
        pred_actions, labels, NUM_CALLS, "none"
    )

    # Compute TP, FP, TN, and FN for each class
    tp = stats[:, 0].squeeze()
    fp = stats[:, 1].squeeze()
    tn = stats[:, 2].squeeze()
    fn = stats[:, 3].squeeze()
    label_counts = stats[:, 4].squeeze()

    # p, r, f1 for each class

    precision = torchmetrics.functional.classification.multiclass_precision(
        pred_actions, labels, NUM_CALLS, "none"
    )
    recall = torchmetrics.functional.classification.multiclass_recall(
        pred_actions, labels, NUM_CALLS, "none"
    )
    f1_score = torchmetrics.functional.classification.multiclass_f1_score(
        pred_actions, labels, NUM_CALLS, "none"
    )
    tick_marks = np.arange(0, 38)
    classes_str = [
        "Pass",
        "Dbl",
        "RDbl",
        "1C",
        "1D",
        "1H",
        "1S",
        "1NT",
        "2C",
        "2D",
        "2H",
        "2S",
        "2NT",
        "3C",
        "3D",
        "3H",
        "3S",
        "3NT",
        "4C",
        "4D",
        "4H",
        "4S",
        "4NT",
        "5C",
        "5D",
        "5H",
        "5S",
        "5NT",
        "6C",
        "6D",
        "6H",
        "6S",
        "6NT",
        "7C",
        "7D",
        "7H",
        "7S",
        "7NT",
    ]

    plt.figure(figsize=(5, 4))
    plt.plot(
        np.arange(0, NUM_CALLS),
        precision,
        color="blue",
        label="precision",
        linestyle="--",
    )
    plt.xlabel("Ordered bids")
    # plt.title("Precision")
    plt.ylim(0, 1)
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.savefig(os.path.join(save_dir, "precision.png"))
    # plt.close()
    # plt.figure()
    plt.plot(np.arange(0, NUM_CALLS), recall, color="red", label="recall")
    plt.xlabel("Ordered calls")
    plt.ylim(0, 1)
    # plt.title("Recall")
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.show()
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "precision_and_recall.svg"), format="svg")
    plt.show()
    plt.close()
    # plt.figure()
    # plt.plot(np.arange(0, NUM_CALLS), f1_score)
    # plt.xlabel("ordered bids")
    # plt.ylabel("f1-score")
    # plt.ylim(0, 1)
    # plt.title("F1-score")
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.show()
    # plt.savefig(os.path.join(save_dir, "f1-score.png"))
    # plt.close()

    # marco p, r, f1
    macro_averaged_precision = (
        torchmetrics.functional.classification.multiclass_precision(
            pred_actions, labels, NUM_CALLS, "macro"
        )
    )
    macro_averaged_recall = torchmetrics.functional.classification.multiclass_recall(
        pred_actions, labels, NUM_CALLS, "macro"
    )
    macro_averaged_f1 = torchmetrics.functional.classification.multiclass_f1_score(
        pred_actions, labels, NUM_CALLS, "macro"
    )

    # micro p, r, f1
    micro_averaged_precision = (
        torchmetrics.functional.classification.multiclass_precision(
            pred_actions, labels, NUM_CALLS, "micro"
        )
    )

    micro_averaged_recall = torchmetrics.functional.classification.multiclass_recall(
        pred_actions, labels, NUM_CALLS, "micro"
    )
    micro_averaged_f1 = torchmetrics.functional.classification.multiclass_f1_score(
        pred_actions, labels, NUM_CALLS, "micro"
    )

    # auc for each class
    auc_per_class = torchmetrics.functional.classification.multiclass_auroc(
        probs, labels, NUM_CALLS, "none"
    )
    # plt.figure()
    # plt.plot(np.arange(0, NUM_CALLS), auc_per_class)
    # plt.xlabel("ordered bids")
    # plt.ylabel("auc")
    # plt.title("AUC")
    # plt.ylim(0.99, 1)
    # # plt.xticks(tick_marks, classes_str, rotation=45)
    # # plt.show()
    # plt.savefig(os.path.join(save_dir, "auc.png"))
    # plt.close()

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
        "confusion_matrix": confusion_matrix,
    }
    # torch.save(stats, os.path.join(save_dir, "stats.pth"))
    pprint.pprint(stats)
    return stats


if __name__ == "__main__":
    test_and_compute_metrics()
    # test()

import copy
import os.path
import pickle
from typing import List, Generator, Tuple

import numpy as np
import torch
from adan import Adan
from tqdm import tqdm, trange

import rl_cpp
import common_utils
from bridge_vars import NUM_CARDS
from nets import PolicyNet2
from bridge_bidding_imitation_learning import cross_entropy, compute_accuracy

OBSERVATION_TENSOR_SIZE = 480 + 38

dataset_dir = "dataset/expert"


def make_sample(dataset: List[List[int]]) -> Generator:
    """
    Generator for making a sample of (observation tensor, label)
    Args:
        dataset: The trajectories list

    Examples:
        sample_gen = make_sample(train_dataset)
        obs, label = next(sample_gen)

    Returns:
        The generator
    """
    while True:
        np.random.shuffle(dataset)
        for trajectory in dataset:
            action_index = np.random.randint(NUM_CARDS, len(trajectory))
            deal = rl_cpp.BridgeDeal()
            deal.cards = trajectory[:NUM_CARDS]
            deal.ddt = np.zeros(20, dtype=int)
            state = rl_cpp.BridgeBiddingState(deal)
            for action in trajectory[NUM_CARDS:action_index]:
                state.apply_action(action - 52)
            observation_tensor = torch.tensor(state.observation_tensor_with_legal_actions())
            yield observation_tensor, trajectory[action_index] - 52


def make_batch(sample_generator: Generator, batch_size: int, device: str = "cuda"):
    """
    Generator for making batch of (observation tensor, label)
    Args:
        device: where the tensors to put
        sample_generator: The sample generator by make_sample()
        batch_size: The batch size

    Examples:
        batch_gen = make_batch(make_sample(train_dataset), batch_size)
        obs, label = next(batch_gen)

    Returns:
        The generator
    """
    observation_tensor = torch.zeros(size=[batch_size, OBSERVATION_TENSOR_SIZE], dtype=torch.float)
    labels = torch.zeros(size=(batch_size,), dtype=torch.long)
    while True:
        for batch_idx in range(batch_size):
            observation_tensor[batch_idx], labels[batch_idx] = next(sample_generator)
        yield observation_tensor.to(device), labels.to(device)


def make_all_samples(dataset: List[List[int]], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    observations = []
    labels = []
    for trajectory in tqdm(dataset):
        deal = rl_cpp.BridgeDeal()
        deal.cards = trajectory[:NUM_CARDS]
        deal.ddt = np.zeros(20, dtype=int)
        state = rl_cpp.BridgeBiddingState(deal)
        for action in trajectory[NUM_CARDS:]:
            observation_tensor = torch.tensor(state.observation_tensor_with_legal_actions())
            label = action - 52
            observations.append(observation_tensor)
            labels.append(label)
            state.apply_action(action - 52)
    return torch.vstack(observations).to(device), torch.tensor(labels).to(device)


def main():
    common_utils.set_random_seeds(42)
    train_batch_size = 16
    num_iterations = 500000
    lr = 3e-4
    eval_freq = 10000
    device = "cuda"
    net = PolicyNet2()
    common_utils.initialize_fc(net)
    net.to(device)
    opt = Adan(params=net.parameters(), lr=lr, fused=True)
    save_dir = common_utils.mkdir_with_increment("imitation_learning")
    saver = common_utils.TopKSaver(10, save_dir, "checkpoint")

    with open(os.path.join(dataset_dir, "valid.pkl"), "rb") as fp:
        valid_data = pickle.load(fp)
    # print(len(valid_data))
    valid_observation, valid_labels = make_all_samples(valid_data, device)
    # print(valid_observation.shape, valid_labels.shape)
    with open(os.path.join(dataset_dir, "train.pkl"), "rb") as fp:
        train_data = pickle.load(fp)
    # print(len(train_data))
    train_generator = make_batch(make_sample(train_data), train_batch_size, device)

    for i_iter in trange(num_iterations):
        opt.zero_grad()
        train_observation, train_labels = next(train_generator)
        log_probs = net(train_observation)
        loss = cross_entropy(log_probs, train_labels, 38)
        loss.backward()
        opt.step()
        if (i_iter + 1) % eval_freq == 0:
            with torch.no_grad():
                log_probs = net(valid_observation)
                valid_loss = cross_entropy(log_probs, valid_labels, 38)
                acc = compute_accuracy(torch.exp(log_probs), valid_labels)
                print(f"iteration {i_iter + 1}, valid loss={valid_loss}, acc={acc}.")
                saver.save(copy.deepcopy(net.state_dict()), acc.item())


if __name__ == '__main__':
    main()

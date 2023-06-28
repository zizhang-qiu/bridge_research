import matplotlib.pyplot as plt
import numpy as np
import torch
import rl_cpp
from agent_for_cpp import SingleEnvAgent
from nets import PolicyNet2
import common_utils
from utils import load_rl_dataset, tensor_dict_to_device

# t0 = torch.tensor([1, 2, 3])
# t1 = torch.tensor([4, 5, 6])
# t2 = torch.tensor([7, 8, 9])
# t3 = torch.tensor([10, 11, 12])
# combined = torch.stack((t0, t1, t2, t3), dim=1)
# print(combined)
# a = torch.tensor([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]])
# print(torch.sum(a, 0))
# print(torch.sum(a, 1))
# manager = rl_cpp.RandomDealManager(1)
# for i in range(10):
#     deal = manager.next()
#     state = rl_cpp.BridgeBiddingState(deal)
#     print(state)
a = torch.tensor([0, 1, 0, 1, 0])
b = torch.nonzero(a).squeeze()
print(b)
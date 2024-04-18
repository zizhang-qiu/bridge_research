import numpy as np
import torch
from torch import nn

from nets import PolicyNet, PolicyNet2
from utils import Evaluator, sl_net
import common_utils


def evaluate_dir(directory: str):
    model_paths = common_utils.find_files_in_dir(directory, "pth", mode=1)
    print(model_paths)
    evaluator = Evaluator(50000, 8, "cuda")
    supervised_net = sl_net()
    net = PolicyNet2()
    for path in model_paths:
        print(path)
        net.load_state_dict(torch.load(path)["model_state_dict"]["policy"])
        avg, sem, *_ = evaluator.evaluate(net, supervised_net)
        print(avg, sem)


def analyze_trained_model(model: PolicyNet2):
    evaluator = Evaluator(50000, 8, "cuda")
    supervised_net = sl_net()
    avg, sem, t, vec_envs0, vec_envs1, imps = evaluator.evaluate(model, supervised_net)
    print(avg, sem)
    envs0 = []
    envs1 = []
    for vec_env in vec_envs0:
        envs0.extend(vec_env.get_envs())
    for vec_env in vec_envs1:
        envs1.extend(vec_env.get_envs())
    lost_games_idx = np.where(np.array(imps) < 0)[0]
    print(lost_games_idx, lost_games_idx.shape)
    for idx in lost_games_idx[:20]:
        print(envs0[idx], envs1[idx], sep='\n')


if __name__ == '__main__':
    #  net1 = PolicyNet2()
    # net1.load_state_dict(torch.load("a2c_fetch/4/folder_10/model9.pth")["model_state_dict"]["policy"])
    # net2 = PolicyNet2()
    # net2.load_state_dict(torch.load("a2c/folder_5/model1.pth")["model_state_dict"]["policy"])
    # evaluator = Evaluator(50000, 8, "cuda")
    # avg, sem, *_ = evaluator.evaluate(net1, net2)
    # print(avg, sem)
    evaluate_dir("a2c/folder_7")

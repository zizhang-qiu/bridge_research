# import math
# import pickle
#
# import matplotlib.pyplot as plt
# import numpy as np

# import common_utils
# from utils import load_rl_dataset, tensor_dict_to_device
# from scipy.stats import norm
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


# from agent_for_cpp import SingleEnvAgent


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
# def calculate_probability(num_boards: int, avg_imp: float, std: float):
#     z = (avg_imp / std) * np.sqrt(num_boards)  # Calculate the z-score
#     probability = norm.cdf(z)  # Calculate the cumulative probability
#     return probability


# print(5.435083 ** 2)
# print(math.sqrt(31.695895))
# n = 10000  # Number of boards
# m = 0.68  # Expected result per board in IMPs
# # s = np.sqrt(31.695895)  # Standard deviation (square root of variance)
# s = 0.05 * math.sqrt(n)
# print(s)
# win_rate = calculate_probability(n, m, s)

# all_contracts = rl_cpp.all_contracts()
# print(all_contracts[420])
# print(all_contracts)
# with open("vs_wbridge5/fetch/log.pkl", "rb") as fp:
#     logs = pickle.load(fp)
# for index in range(1600, 10000):
#     if logs[index] == "":
#         print(index)
#         break
# imps0 = np.load("vs_wbridge5/imps.npy")
# imps1 = np.load("vs_wbridge5/fetch/imps.npy")
# imps = np.concatenate([imps0[:500], imps1[500:1712]])
# print(imps.shape)
# print(common_utils.get_avg_and_sem(imps))
# np.save("vs_wbridge5/imps.npy", imps)
def is_row_in_2d_array(arr_2d, arr_1d):
    """
    Check if a 1D array is a row of the 2D array.

    Parameters:
        arr_2d (numpy.ndarray): The 2D numpy array to search for the row.
        arr_1d (numpy.ndarray): The 1D numpy array to check if it is a row.

    Returns:
        bool: True if the 1D array is a row in the 2D array, False otherwise.
    """
    return np.any(np.all(arr_2d == arr_1d, axis=1))


# with open(r"E:\baidunetdiskdownload\log(2).pkl", "rb") as fp:
#     logs0to499 = pickle.load(fp)
#
# with open(r"E:\baidunetdiskdownload\log.pkl", "rb") as fp:
#     logs500to1712 = pickle.load(fp)
#
# logs0to1712: List[str] = logs0to499[:500] + logs500to1712[500:1712]
#
classes_str = [
    "Pass",
    "Dbl",
    "RDbl",
    "1C",
    "1D",
    "1H",
    "1S",
    "1N",
    "2C",
    "2D",
    "2H",
    "2S",
    "2N",
    "3C",
    "3D",
    "3H",
    "3S",
    "3N",
    "4C",
    "4D",
    "4H",
    "4S",
    "4N",
    "5C",
    "5D",
    "5H",
    "5S",
    "5N",
    "6C",
    "6D",
    "6H",
    "6S",
    "6N",
    "7C",
    "7D",
    "7H",
    "7S",
    "7N",
]


def extract_bids_from_str(s: str) -> List[str]:
    words = [word for word in s.split() if word.strip()]
    bid_strs = []
    for word in words:
        if word in classes_str:
            bid_strs.append(word)
    return bid_strs


trump_strs = ["C", "D", "H", "S", "N"]


def bid_str_2_action(bid_str: str):
    if bid_str == "Pass":
        return 0
    if bid_str == "Dbl":
        return 1
    if bid_str == "RDbl":
        return 2
    bid_level = int(bid_str[0])
    bid_trump = trump_strs.index(bid_str[1])
    action = 3 + (bid_level - 1) * 5 + bid_trump
    assert 3 <= action < 38
    return action


def bidding_history_from_state_str(state_str: str) -> List[int]:
    lines = state_str.split("\n")
    # print(lines)
    print(lines[15:])
    bid_strs = []
    for line in lines[15:]:
        bid_strs += extract_bids_from_str(line)
    # print(bid_strs)
    actions = [bid_str_2_action(bid_str) for bid_str in bid_strs]
    # print(actions)
    return actions


# dataset = load_rl_dataset("vs_wb5_open_spiel")
#
# imp_results = []
# for i, log in enumerate(logs0to1712):
#     print(i)
#     print(log)
#     open_str = log[log.find("Vul"):min(log.find("Declarer"), log.find("close") - 1)]
#     close_str = log[log.rfind("Vul"):len(log) if log.rfind("Declarer") < log.rfind("Vul") else log.rfind("Declarer")]
#     print(f"open_str:{open_str}", f"close_str:{close_str}", sep="\n")
#     open_bid_history = bidding_history_from_state_str(open_str)
#     close_bid_history = bidding_history_from_state_str(close_str)
#     print(open_bid_history, close_bid_history)
#     deal = rl_cpp.BridgeDeal()
#     deal.cards = dataset["cards"][i]
#     deal.ddt = dataset["ddts"][i]
#     open_state = rl_cpp.BridgeBiddingState(deal)
#     for a in open_bid_history:
#         open_state.apply_action(a)
#     close_state = rl_cpp.BridgeBiddingState(deal)
#     for a in close_bid_history:
#         close_state.apply_action(a)
#     res = IMPResult(dataset["cards"][i], dataset["ddts"][i], open_state, close_state)
#     imp_results.append(res)


# with open("imp_result.pkl", "rb") as fp:
#     imp_res: IMPResult = pickle.load(fp)
# print(imp_res)
# print(imp_res.cards)
# print(imp_res.ddt)
# print(imp_res.open_state)
# print(imp_res.close_state)
# print(imp_res.open_score)
# print(imp_res.close_score)
# print(imp_res.imp)
#
# with open("imp_results_rl_bmcs.pkl", "rb") as fp:
#     imp_results: List[IMPResult] = pickle.load(fp)
# # print(imp_results)
# imps = []
# for res in imp_results:
#     imps.append(res.imp)
# print(common_utils.get_avg_and_sem(imps))
#
# opening_stats = []
# for res in imp_results:
#     open_state = res.open_state
#     bid, hand_evaluation, player = open_state.opening_bid_and_hand_evaluation()
#     if player in [0, 2] and bid != 0:
#         opening_stats.append((bid, hand_evaluation, player))
#
#     close_state = res.close_state
#     bid, hand_evaluation, player = close_state.opening_bid_and_hand_evaluation()
#     if player in [1, 3] and bid != 0:
#         opening_stats.append((bid, hand_evaluation, player))
#
# print(len(opening_stats))
# opening_counter = Counter(opening_stats)
#
# losses = np.load("data/belief_loss.npy")
# plt.plot(np.arange(0, 21), losses[:21])
# plt.xticks(np.arange(0, 21, 5))
# plt.xlabel("Bidding Length", fontsize=12)
# plt.ylabel("Average Cross Entropy Loss", fontsize=12)
# plt.show()


# plt.savefig("belief_quality.svg", format="svg")

# p_net = PolicyNetwork3(4, 2048, use_layer_norm=True)
# print(p_net)
# matplotlib.rcParams['text.usetex'] = True
# imps = [0.25, 0.41, 0.63, 0.57, 0.85, 0.68, 0.93]
# imps.reverse()
#
# # plt.figure(figsize=(5, 3))
#
# fig, ax = plt.subplots(figsize=(6, 4))
#
# ax.bar(
#     np.arange(7),
#     imps,
#     color=[
#         "steelblue",
#         "steelblue",
#         "deepskyblue",
#         "deepskyblue",
#         "seagreen",
#         "darkorange",
#         "tomato",
#     ],
#     yerr=[0.05, 0.05, 0.05, 0.05, 0.22, 0.27, 0],
#     error_kw={"ecolor": "black", "capsize": 3},
# )
# ax.set_ylim(0.0, 1.0)
# ax.set_ylabel("IMPs per deal")
# ax.set_xticks(
#     np.arange(7),
#     labels=[
#         "RL-BMCS\n(Ours)",
#         "RL\n(Ours)",
#         "PI+Search",
#         "PI",
#         "JPS",
#         "Simple",
#         "DNNs",
#     ],
#     fontsize=10,
# )
# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# # plt.savefig("1.eps", format="eps")
# plt.show()

# imps = [0.9322, 0.80, 0.6776, 0.5883, 0.5311]
# ax.bar(
#     np.arange(5),
#     imps,
#     color=["steelblue", "deepskyblue", "steelblue", "deepskyblue", "deepskyblue"],
#     yerr=[0.05, 0.05, 0.049, 0.0485, 0.05],
#     error_kw={"ecolor": "black", "capsize": 3},
# )
# ax.set_ylim(0.0, 1.0)
# ax.set_ylabel("IMPs per deal")
# ax.set_xticks(
#     np.arange(5), labels=["RL-BMCS", "RL-MCS", "RL", "No HI", "No PR"], fontsize=10
# )
# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# plt.show()

# a = [
#     42.85905456542969,
#     41.99424362182617,
#     41.489078521728516,
#     41.045345306396484,
#     41.183685302734375,
#     40.85429382324219,
#     40.689449310302734,
#     40.38103103637695,
#     40.46717834472656,
#     40.20991516113281,
#     40.17684555053711,
#     40.05743408203125,
#     40.010311126708984,
#     39.81943893432617,
#     39.661537170410156,
#     39.4807243347168,
#     39.398048400878906,
#     39.29787826538086,
#     39.34328842163086,
#     39.192806243896484,
#     38.97060012817383,
# ]
# b = [
#     42.86143206787109,
#     43.659655307769775,
#     44.3288607749939,
#     44.82969024658203,
#     44.65773942184448,
#     45.11836152267456,
#     45.30268706893921,
#     45.63121628570557,
#     45.58002278900147,
#     46.02051647186279,
#     46.05760515975952,
#     46.37699789047241,
#     46.440963092803955,
#     46.83868597412109,
#     47.18840441513061,
#     47.614603897094725,
#     47.83884443664551,
#     48.276506340026856,
#     48.348668472290036,
#     48.849986911773684,
#     48.95163544845581,
# ]
#
# plt.figure()
# plt.plot(np.arange(21), a, label="belief")
# plt.plot(np.arange(21), b, label="random")
# plt.xlabel("Bidding length")
# plt.ylabel("Cross entropy")
# plt.xticks(np.arange(21))
# plt.legend()
# plt.show()


# p_net = PolicyNet2()
# print(count_model_parameters(p_net))
def make_multi_hot(total: int, num_hot: int) -> torch.Tensor:
    t = torch.zeros(total, dtype=torch.float)
    indices = torch.multinomial(
        torch.arange(total, dtype=torch.float), num_hot, replacement=False
    )
    t[indices] = 1
    return t


# torch.manual_seed(2)
# label = make_multi_hot(156, 39)
#
# num_samples = 1000
# pred = torch.zeros([num_samples, 156], dtype=torch.float)
# for i in range(num_samples):
#     pred[i] = make_multi_hot(156, 39)
#
# loss = -torch.log(pred + 1e-15) * label
# print(loss.sum(1).mean())

belief_loss = [
    42.85734939575195,
    42.092506408691406,
    41.498023986816406,
    41.16340637207031,
    41.198551177978516,
    41.00836181640625,
    40.630271911621094,
    40.568199157714844,
    40.54984664916992,
    40.4505500793457,
    40.11854553222656,
    40.13051223754883,
    39.97572708129883,
    39.8709716796875,
    39.66355514526367,
    39.650203704833984,
    39.290687561035156,
    39.289764404296875,
    39.149295806884766,
    39.18234634399414,
    38.96243667602539,
]
random_loss = [
    897.8812222290039,
    898.0967465820313,
    898.0995770263672,
    897.9678473510742,
    898.0255612182617,
    898.041139465332,
    897.8191220703125,
    898.1713812866211,
    898.0855874633789,
    898.1358772583008,
    897.9741666259765,
    897.7989517822266,
    898.0561595458985,
    898.0007980346679,
    897.9745812988281,
    898.0561264038085,
    898.0995443725586,
    897.9384535522461,
    897.8428849487304,
    898.1621622924805,
    898.1638201904296,
]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(random_loss, label="Uniform Sample", color="orange", linestyle="--")
ax2.plot(belief_loss, label="Belief Network")
ax1.set_ylim(896.5, 900)
ax2.set_ylim(38, 43.5)
ax1.spines.top.set_visible(False)
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax2.spines.right.set_visible(False)
ax1.xaxis.set_ticks_position("none")
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax2.set_xticks(range(0, len(belief_loss), 5))

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
# ax1.legend()
# ax2.legend()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")
# plt.legend()
ax2.set_xlabel("Bidding Length")
# ax1.set_ylabel("Average Cross Entropy Loss", loc="bottom")
plt.text(
    -3.5, 40, "Average Cross Entropy Loss", fontdict={"fontsize": 13, "rotation": 90}
)
plt.show()

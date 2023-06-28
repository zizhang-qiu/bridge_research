import pickle

import numpy as np
from matplotlib import pyplot as plt

from bridge_consts import NUM_DENOMINATIONS, NUM_BIDS, DENOMINATIONS

all_contract_str = [str(level) + trump for level in range(1, 8) for trump in DENOMINATIONS]


def main():
    with open("analyze.pkl", "rb") as fp:
        deal_info = pickle.load(fp)
    # print(deal_info)
    trumps_count = np.zeros(NUM_DENOMINATIONS)
    contract_count = np.zeros(NUM_BIDS)
    trick_difference_per_contract = np.zeros(NUM_BIDS)
    for info in deal_info:
        trump, actual_tricks, dd_tricks = info
        trumps_count[trump] += 1
        contract = ((dd_tricks - 6) - 1) * NUM_DENOMINATIONS + trump
        contract_count[contract] += 1
        trick_difference = dd_tricks - actual_tricks
        trick_difference_per_contract[contract] += trick_difference

    print(trumps_count)
    print(trick_difference_per_contract)
    print(trick_difference_per_contract / contract_count)
    plt.bar(np.arange(NUM_BIDS), trick_difference_per_contract / contract_count)
    plt.xticks(np.arange(NUM_BIDS), labels=all_contract_str)
    plt.show()


if __name__ == '__main__':
    main()

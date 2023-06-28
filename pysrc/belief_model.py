from typing import Final, Dict, List, Any

import torch
from torch import nn as nn
import torch.nn.functional as F
import rl_cpp
import common_utils


class BeliefModel(nn.Module):
    def __init__(self):
        """
        A policy net to output policy distribution.
        """
        super(BeliefModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 156)
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        policy = F.sigmoid(out)
        return policy

    @torch.jit.export
    def sample(self, obs: Dict[str, torch.Tensor], current_player: int, num_sample: int):
        s = obs["s"]
        assert s.dim() == 1
        pred = self.forward(s.unsqueeze(0)).squeeze().cpu()
        own_cards = s[-52:]
        own_cards = own_cards.cpu()
        sampled_cards = torch.zeros([num_sample, 52], dtype=torch.int)
        basic_indices = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
        for i_sample in range(num_sample):
            cards_mask = 1 - own_cards
            cards = torch.zeros(52, dtype=torch.int)
            cards.scatter_(0, basic_indices + current_player, torch.nonzero(own_cards).squeeze().int())
            # cards_per_player = torch.jit.annotate(List[torch.Tensor], [])
            # for j in range(4):
            #     cards_per_player.append(torch.tensor([]))
            # cards_per_player[current_player] = torch.nonzero(own_cards).squeeze()
            for i in range(3):
                next_player_cards_pred = pred[i * 52: (i + 1) * 52].clone()
                next_player_cards_pred *= cards_mask
                next_player_sample_cards = torch.multinomial(next_player_cards_pred, 13, False)
                cards.scatter_(0, basic_indices + ((current_player + (i + 1)) % 4), next_player_sample_cards.int())
                # cards_per_player[(current_player + (i + 1)) % 4] = next_player_sample_cards
                # print(next_player_sample_cards)
                next_player_sample_cards_multi_hot = common_utils.multi_hot_1d(next_player_sample_cards, 52)
                cards_mask *= 1 - next_player_sample_cards_multi_hot
            # make deal
            # cards = torch.stack(cards_per_player, 1).flatten()
            sampled_cards[i_sample] = cards
        return {"cards": sampled_cards}

    def compute_loss(self, batch: rl_cpp.ObsBelief):
        pred = self.forward(batch.obs["s"])
        loss = F.binary_cross_entropy(pred, batch.belief["belief"])
        return loss.mean()


class ARBeliefModel(nn.Module):
    device: Final[str]
    input_key: Final[str]
    ar_input_key: Final[str]
    ar_target_key: Final[str]
    in_dim: Final[int]
    hid_dim: Final[int]
    out_dim: Final[int]
    hand_size: Final[int]
    num_lstm_layer: Final[int]
    num_sample: Final[int]
    fc_only: Final[bool]

    def __init__(
            self, device, in_dim, hid_dim, hand_size, out_dim, num_sample, fc_only
    ):
        """
        mode: priv: private belief prediction
              publ: public/common belief prediction
        """
        super().__init__()
        self.device = device
        self.input_key = "priv_s"
        self.ar_input_key = "own_hand_ar_in"
        self.ar_target_key = "own_hand"

        self.in_dim = in_dim
        self.hand_size = hand_size
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.num_sample = num_sample
        self.fc_only = fc_only

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.emb = nn.Linear(25, self.hid_dim // 8, bias=False)
        self.auto_regress = nn.LSTM(
            self.hid_dim + self.hid_dim // 8,
            self.hid_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        self.auto_regress.flatten_parameters()

        self.fc = nn.Linear(self.hid_dim, self.out_dim)

    def get_h0(self, batch_size: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batch_size, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid


if __name__ == '__main__':
    model = ARBeliefModel("cuda", 480, 512, 13 * 3, 52 * 3, 0, False)
    model.to("cuda")
    h0 = model.get_h0(10)
    print(h0)
    print(h0["h0"].shape, h0["c0"].shape)

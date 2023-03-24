"""
@file:bluechip_bridge
@author:qzz
@date:2023/3/4
@encoding:utf-8
"""
import abc
import torch
from src.bridge.bridge_vars import NUM_PLAYERS, NUM_CARDS
from src.common_utils.assert_utils import assert_in_range
import re
import socket
import sys
import time
from typing import Callable, Union, Optional, List
import rl_cpp

# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Wraps third-party bridge bots to make them usable in OpenSpiel.

This code enables OpenSpiel interoperation for bots which implement the BlueChip
bridge protocol. This is widely used, e.g. in the World computer bridge
championships. For a rough outline of the protocol, see:
http://www.bluechipbridge.co.uk/protocol.htm

No formal specification is available. This implementation has been verified
to work correctly with WBridge5.

This bot controls a single player in the full game of bridge, including both the
bidding and play phase. It chooses its actions by invoking an external bot which
plays the full game of bridge. This means that each time the bot is asked for an
action, it sends up to three actions (one for each other player) to the external
bridge bot, and obtains an action in return.
"""

# Example session:
#
# Recv: Connecting "WBridge5" as ANYPL using protocol version 18
# Send: WEST ("WBridge5") seated
# Recv: WEST ready for teams
# Send: Teams: N/S "silent" E/W "bidders"
# Recv: WEST ready to start
# Send: Start of board
# Recv: WEST ready for deal
# Send: Board number 8. Dealer WEST. Neither vulnerable.
# Recv: WEST ready for cards
# Send: WEST's cards: S A T 9 5. H K 6 5. D Q J 8 7 6. C 7.
# Recv: WEST PASSES
# Recv: WEST ready for  NORTH's bid
# Send: EAST PASSES
# Recv: WEST ready for EAST's bid
# Send: EAST bids 1C
# Recv: WEST ready for  SOUTH's bid


# Template regular expressions for messages we receive
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_PLAYER_ACTION = ("(?P<seat>NORTH|SOUTH|EAST|WEST) "
                  "((?P<pass>PASSES)|(?P<dbl>DOUBLES)|(?P<rdbl>REDOUBLES)|bids "
                  "(?P<bid>[^ ]*)|(plays (?P<play>[23456789tjqka][cdhs])))"
                  "(?P<alert> Alert.)?")
_READY_FOR_OTHER = ("{seat} ready for "
                    "(((?P<other>[^']*)'s ((bid)|(card to trick \\d+)))"
                    "|(?P<dummy>dummy))")

# Templates for fixed messages we receive
_READY_FOR_TEAMS = "{seat} ready for teams"
_READY_TO_START = "{seat} ready to start"
_READY_FOR_DEAL = "{seat} ready for deal"
_READY_FOR_CARDS = "{seat} ready for cards"
_READY_FOR_BID = "{seat} ready for {other}'s bid"

# Templates for messages we send
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "north-south" E/W "east-west"'
_START_BOARD = "start of board"
_DEAL = "Board number {board}. Dealer NORTH. Neither vulnerable."
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_ACTION = "{player} {action}"
_PLAYER_TO_LEAD = "{seat} to lead"
_DUMMY_CARDS = "Dummy's cards: {}"

# BlueChip bridge protocol message constants
_SEATS = ["NORTH", "EAST", "SOUTH", "WEST"]
_TRUMP_SUIT = ["C", "D", "H", "S", "NT"]
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_SUIT = _TRUMP_SUIT[:4]
_NUMBER_SUITS = len(_SUIT)
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_LSUIT = [x.lower() for x in _SUIT]
_LRANKS = [x.lower() for x in _RANKS]

# action ids
_ACTION_PASS = 0
_ACTION_DBL = 1
_ACTION_RDBL = 2


class Controller(abc.ABC):
    """An abstract class for bluechip bot controller"""

    @abc.abstractmethod
    def send_line(self, line: str):
        ...

    @abc.abstractmethod
    def read_line(self) -> str:
        ...

    @abc.abstractmethod
    def terminate(self):
        ...


def _expect_regex(controller: Controller, regex: str):
    """Reads a line from the controller, parses it using the regular expression."""
    line = controller.read_line()
    match = re.match(regex, line)
    if not match:
        raise ValueError("Received '{}' which does not match regex '{}'".format(
            line, regex))
    return match.groupdict()


def _expect(controller: Controller, expected: str):
    """Reads a line from the controller, checks it matches expected line exactly."""
    line = controller.read_line()
    if expected != line:
        raise ValueError("Received '{}' but expected '{}'".format(line, expected))


def _hand_string(cards: List[int]):
    """Returns the hand of the to-play player in the state in BlueChip format."""
    if (n := len(cards)) != 13:
        raise ValueError(f"Must have 13 cards, but the cards provided is {n}")
    suits = [[] for _ in range(4)]
    for card in reversed(sorted(cards)):
        suit = card % 4
        rank = card // 4
        suits[suit].append(_RANKS[rank])
    for i in range(4):
        if suits[i]:
            suits[i] = _TRUMP_SUIT[i] + " " + " ".join(suits[i]) + "."
        else:
            suits[i] = _TRUMP_SUIT[i] + " -."
    return " ".join(suits)


def _connect(controller: Controller, seat: str):
    """Performs the initial handshake with a BlueChip bot."""
    client_name = _expect_regex(controller, _CONNECT)["client_name"]
    controller.send_line(_SEATED.format(seat=seat, client_name=client_name))
    _expect(controller, _READY_FOR_TEAMS.format(seat=seat))
    controller.send_line(_TEAMS)
    _expect(controller, _READY_TO_START.format(seat=seat))


def _new_deal(controller: Controller, seat: str, hand: str, board: str):
    """Informs a BlueChip bots that there is a new deal."""
    controller.send_line(_START_BOARD)
    _expect(controller, _READY_FOR_DEAL.format(seat=seat))
    controller.send_line(_DEAL.format(board=board))
    _expect(controller, _READY_FOR_CARDS.format(seat=seat))
    controller.send_line(_CARDS.format(seat=seat, hand=hand))


# TODO: Rethink the bot
class BlueChipBridgeBot:
    """An OpenSpiel bot, wrapping a BlueChip bridge bot implementation."""

    def __init__(self, player_id: int, controller_factory: Callable[[int], Controller], port: int):
        """
        Initialization
        Args:
            player_id(int):the bot's seat, 0 for north
            controller_factory: the function to create controller
        """
        assert_in_range(player_id, 0, NUM_PLAYERS)
        self._player_id = player_id
        self._controller_factory = controller_factory
        self._seat = _SEATS[player_id]
        self._num_actions = NUM_CARDS
        self._board = 0
        self._port = port
        self._internal_state: Optional[rl_cpp.BridgeBiddingState] = None
        self._controller: Optional[Controller] = None

    def player_id(self):
        return self._player_id

    def restart(self):
        """Indicates that we are starting a new episode."""

        self._num_actions = NUM_CARDS
        # We didn't see the end of the episode, so the external bot will still
        # be expecting it. If we can autoplay other people's actions to the end
        # (e.g. everyone passes or players play their last card), then do that.
        # if not self._env.terminated():
        #     state = self._state.clone()
        #     while (not state.terminated()
        #            and state.current_player() != self._player_id):
        #         legal_actions = state.legal_actions()
        #         if _ACTION_PASS in legal_actions:
        #             state.apply(_ACTION_PASS)
        #         elif len(legal_actions) == 1:
        #             state.apply_action(legal_actions[0])
        #     if state.terminated():
        #         self.inform_state(state)
        # Otherwise, we will have to restart the external bot, because
        # the protocol makes no provision for this case.
        if self._internal_state is not None:
            if not self._internal_state.terminated():
                if self._controller is not None:
                    self._controller.terminate()
                    self._controller = None

    def inform_action(self, state, player, action):
        del player, action
        self.inform_state(state)

    def inform_state(self, state: rl_cpp.BridgeBiddingState):
        # Connect if we need to.
        if self._controller is None:
            self._controller = self._controller_factory(self._port)
            _connect(self._controller, self._seat)
        full_history = state.history()
        if len(full_history) == NUM_CARDS + self._player_id:
            self._internal_state = rl_cpp.BridgeBiddingState(0, full_history[:NUM_CARDS], False, False,
                                                             state.get_double_dummy_table())
            self._update_for_state()

        known_history = self._internal_state.history()
        if full_history[:len(known_history)] != known_history:
            raise ValueError(
                "Supplied state is inconsistent with bot's internal state\n"
                f"Supplied state:\n{state}\n"
                f"Internal state:\n{self._internal_state}\n")

        for action in full_history[len(known_history):]:
            self._internal_state.apply_action(action)
            if not self._internal_state.terminated():
                self._update_for_state()

    def _update_for_state(self):
        """Called for all non-chance nodes, whether we have to act."""
        # Get the actions in the game so far.
        actions = self._internal_state.history()

        # If this is the first time we've seen the deal, send our hand.
        if len(actions) == NUM_CARDS:
            self._board += 1
            _new_deal(self._controller, self._seat,
                      rl_cpp.get_hand_string(actions[self._player_id:52:4]), str(self._board))

        # Send actions since last `step` call.
        for other_player_action in actions[self._num_actions:]:
            other = _expect_regex(self._controller,
                                  _READY_FOR_OTHER.format(seat=self._seat))
            other_player = other["other"]
            self._controller.send_line(
                _OTHER_PLAYER_ACTION.format(
                    player=other_player,
                    action=rl_cpp.bid_action_to_str(other_player_action)))
        self._num_actions = len(actions)

        # If the episode is terminal, send (fake) timing info.
        # if self._env.terminated():
        # self._controller.send_line(
        #     "Timing - N/S : this board  [1:15],  total  [0:11:23].  "
        #     "E/W : this board  [1:18],  total  [0:10:23]"
        # )

    def step(self, state: rl_cpp.BridgeBiddingState):
        """Returns an action for the given state."""
        # Bring the external bot up-to-date.
        self.inform_state(state)

        # Get our action from the bot.
        our_action = _expect_regex(self._controller, _PLAYER_ACTION)
        self._num_actions += 1
        if our_action["pass"]:
            return _ACTION_PASS
        elif our_action["dbl"]:
            return _ACTION_DBL
        elif our_action["rdbl"]:
            return _ACTION_RDBL
        elif our_action["bid"]:
            return rl_cpp.bid_str_to_action(our_action["bid"])

    def terminate(self):
        self._controller.terminate()
        self._controller = None


if __name__ == '__main__':
    # print(rl_cpp.bid_action_to_str(0))
    # print(rl_cpp.bid_str_to_action("1NT"))
    # s = "EAST bids 1C"
    # match = re.match(_PLAYER_ACTION, s)
    # print(match.groupdict())
    pass

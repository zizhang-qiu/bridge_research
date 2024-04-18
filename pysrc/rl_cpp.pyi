from typing import List, overload, Tuple, NoReturn, Union, Optional, Dict

import numpy as np
import torch

Player = int
Action = int
TensorDict = Dict[str, torch.Tensor]
CardsLike = Union[np.ndarray, List[Action]]
DDTLike = Union[np.ndarray, List[Action]]

class ModelLocker:
    def __init__(self, py_models: List[torch.jit.ScriptModule], device: str):
        """
        The ModelLocker is a class which locks a list of torch.jit.ScriptModule
        and can be used in c++ actors
        Args:
            py_models: A list of ScriptModule
            device: The device the models at, used for actors to move the obs to correct tensor
        """
        ...
    def update_model(self, py_model: torch.jit.ScriptModule) -> NoReturn:
        """
        Update the latest model's parameter using a ScriptModule
        Args:
            py_model: A ScriptModule provides parameters

        """
        ...

class SingleEnvActor:
    def __init__(self, model_locker: ModelLocker):
        """
        An actor acts for a single environment
        Args:
            model_locker: The model locker saves ScriptModule to call
        """
        ...
    def act(self, obs: TensorDict) -> TensorDict:
        """
        Get action for given obs tensor
        Args:
            obs: The obs

        Returns:
            The reply
        """
        ...
    def get_top_k_actions_with_min_prob(
        self, obs: TensorDict, k: int, min_prob: float
    ) -> TensorDict: ...
    def get_prob_for_action(self, obs: TensorDict, action: Action) -> float: ...

class VecEnvActor:
    def __init__(self, model_locker: ModelLocker):
        """
        An actor acts for vectorized environment
        Args:
            model_locker: The model locker saves ScriptModule to call
        """
        ...
    def act(self, obs: TensorDict) -> TensorDict:
        """
        Get action for given obs tensor
        Args:
            obs: The obs

        Returns:
            The reply
        """
        ...

class ReplayBuffer:
    def __init__(
        self,
        state_size: int,
        num_actions: int,
        capacity: int,
        alpha: float,
        eps: float,
        beta: float,
    ):
        """
        A buffer stores states(obs), actions, rewards and log_probs
        Args:
            state_size: The size of state tensor
            num_actions: The number of actions
            capacity: The capacity of the buffer
        """
        ...
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        log_probs: torch.Tensor,
    ): ...
    def sample(
        self, batch_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of obs, actions, rewards and log_probs
        Args:
            device: The device
            batch_size(int): Sample batch size

        Returns:
            Tuple of obs, actions, rewards and log_probs
        """
        ...
    def size(self) -> int:
        """
        Get the size of replay buffer
        Returns:
            The size of buffer, which is the amount of items
        """
        ...
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all items in the buffer
        Returns:
            The tuple of all obs, actions, rewards and log probs
        """
        ...
    def num_add(self) -> int:
        """
        Get the number of items added to buffer
        Returns:
            The number of add
        """
        ...

class BridgeTransitionBuffer:
    def __init__(self): ...
    def push_obs_and_reply(self, obs: TensorDict, action: TensorDict): ...
    def push_to_replay_buffer(
        self, replay_buffer: ReplayBuffer, final_reward: float
    ): ...
    def pop_transitions(self) -> Tuple[List[Transition], torch.Tensor]: ...

class MultiAgentTransitionBuffer:
    def __init__(self, num_agents: int): ...
    def push_obs_and_reply(
        self, player: Player, obs: TensorDict, reply: TensorDict
    ): ...
    def push_to_replay_buffer(
        self, replay_buffer: ReplayBuffer, final_reward: Union[List[float], np.ndarray]
    ): ...

class BridgeDeal:
    def __init__(self):
        self.cards: CardsLike = ...
        self.dealer: Player = ...
        self.is_dealer_vulnerable: bool = ...
        self.is_non_dealer_vulnerable: bool = ...
        self.ddt: Optional[DDTLike] = ...
        self.par_score: Optional[int] = ...

class DealManager: ...

class BridgeDealManager(DealManager):
    def __init__(self, cards: CardsLike, ddts: DDTLike):
        """
        A manager stores deals.
        Args:
            cards: A list or ndarray of cards.
            ddts: A list or ndarray of ddts.
        """
        ...
    def size(self) -> int:
        """Get the size og manager."""
        ...
    def next(self) -> BridgeDeal:
        """Get next deal of manager."""
        ...

class Contract:
    level: int = ...
    declarer: Player = ...

    def trumps(self) -> int: ...
    def double_status(self) -> int: ...

class HandEvaluation:
    high_card_points: int = ...
    length_points: int = ...
    shortness_points: int = ...
    support_points: int = ...
    control_count: int = ...
    length_per_suit: List[int] = ...

class BridgeBiddingState:
    def __init__(self, deal: BridgeDeal):
        """
        A states records bridge bidding. The players are represented as 0,1,2,3 for NESW.
        The cards are represented as rank * NumSuits + suit, i.e. 0=2C, 1=2D, 2=2H, 3=2S, etc.
        Args:
            deal: The BridgeDeal which should contain
                dealer: The player to start bidding
                cards: A array of cards to deal, the cards are dealt clockwise,
                    i.e. cards[0] gives to north, cards[1] gives to East
                is_dealer_vulnerable: whether the dealer size is vulnerable
                is_non_dealer_vulnerable: whether the non-dealer side is vulnerable
                ddt: The double dummy table, which is a 20-element array. The array should be ordered as
                    C(NESW)-D-H-S-NT. An example:
                        C	D	H	S	NT

                    N	0	0	10	9	0

                    E	12	12	3	4	8

                    S	0	0	10	9	0

                    W	12	12	3	4	8

                    the ddt array should be [ 0 12  0 12  0 12  0 12 10  3 10  3  9  4  9  4  0  8  0  8]
        """
        ...
    def history(self) -> List[Action]:
        """
        Get history of actions so far, include deal actions
        Returns:
            The list of actions
        """
        ...
    def bid_str(self) -> str:
        """
        Get a string of bidding sequence
        Returns:
            The bidding string, start with the dealer, e.g. Dealer is N, 1C, Pass, Pass, Pass
        """
        ...
    def bid_history(self) -> List[Action]:
        """
        Get the history action of only bidding phase
        Returns:
            The list of bidding actions
        """
        ...
    def bid_str_history(self) -> List[str]:
        """
        Get the string of each bid in history
        Returns:
            The list of bid strings, e.g. ['1C', 'Pass', 'Pass', 'Pass']
        """
        ...
    def apply_action(self, action: Action):
        """
        Apply an action to the state, should not be called at terminal
        Args:
            action: the bid action. Bid is a number in range [0, 38).
                The first 3 action is 0->Pass, 1->Double, 2->ReDouble,
                and other calls is represented as 3->1C, 4->1D, ... 36->7S, 37->7N

        Returns:

        """
        ...
    def terminated(self) -> bool:
        """
        Get the terminal information
        Returns:
            A boolean value of whether the state is terminated
        """
        ...
    def returns(self) -> List[float]:
        """
        Get duplicate scores, can only be called if terminated
        Returns:
            A list of duplicate scores, ordered from North clockwise, e.g. [50.0, -50.0, 50.0, -50.0]
        """
        ...
    def contract_str(self) -> str:
        """
        Get a string of contract.
        Returns:
            The contract string, the string contains the contract, the declarer and whether doubled, e.g. 7NXX N.
        """
        ...
    def current_player(self) -> Player:
        """
        Get current player's seat
        Returns:
            The player id. 0 for North, clockwise.
        """
        ...
    def current_phase(self) -> int:
        """
        Get current phase.
        Returns:
            The phase, 0 for auction, 1 for game over.
        """
        ...
    def get_double_dummy_table(self) -> List[int]:
        """
        Get the double dummy table in 1d array
        Returns:
            The double dummy table.
        """
    def get_contract(self) -> Contract: ...
    def observation_tensor(self, player: Optional[Player]) -> List[float]: ...
    def hidden_observation_tensor(self) -> List[float]: ...
    def terminate(self) -> NoReturn:
        """
        Force terminate the state.
        Returns:
            No returns.
        """
        ...
    def legal_actions(self) -> List[Action]:
        """
        Get legal actions.
        Returns:
            A list of actions which are legal at this time point.
        """
        ...
    def clone(self) -> BridgeBiddingState:
        """
        Get a cloned state.
        Returns:
            The cloned state.
        """
        ...
    def get_hand_evaluation(self) -> List[HandEvaluation]: ...
    def observation_tensor_with_hand_evaluation(self) -> List[float]: ...
    def __repr__(self) -> str: ...
    def observation_tensor_with_legal_actions(self) -> List[float]: ...
    def get_partner_cards(self) -> List[int]: ...
    def cards_tensor(self) -> List[float]: ...
    def final_observation_tensor(self) -> List[float]: ...
    def score_for_contracts(
        self, player: Player, contracts: List[int]
    ) -> List[int]: ...
    def get_player_cards(self, player: Player) -> List[int]:
        """
        Get a player's cards.
        Args:
            player: The player.

        Returns:
            The cards of the player.
        """
        ...
    def opening_bid_and_hand_evaluation(self) -> Tuple[int, HandEvaluation, int]: ...
    def num_declarer_tricks(self) -> int:
        """Return the number of tricks won by the declarer side"""

class BridgeBiddingEnv:
    def __init__(self, deal_manager: BridgeDealManager, greedy: List[int]):
        """
        A Bridge Bidding Environment.
        Args:
            deal_manager: The deal manager stores deals.
            greedy: Whether greedy for four players. e.g. [1,1,1,1] means all the players are greedy.
        """
        ...
    def reset(self) -> TensorDict:
        """
        Reset the state and return an initial obs. Should be called at first of each game.
        Returns:
            The initial obs
        """
        ...
    def step(self, reply: TensorDict):
        """
        Make one step in environment and get next obs, reward and terminal signal.
        Args:
            reply: The reply from actor.
        """
        ...
    def returns(self) -> List[float]:
        """
        Get duplicate scores for each player. Should be called after the env terminated.
        Returns:
            The list of duplicate scores.
        """
        ...
    def terminated(self) -> bool:
        """
        Whether the env is terminated.
        Returns:
            A boolean value of terminal.
        """
        ...
    def get_state(self) -> BridgeBiddingState:
        """
        Get the state.
        Returns:
            The BridgeBiddingState.
        """
        ...
    def get_num_deals(self) -> int:
        """
        Get the number of deals played in the env.
        Returns:
            The number of deals.
        """
        ...
    def get_feature_size(self) -> int: ...
    def get_feature(self) -> TensorDict: ...
    def __repr__(self) -> str: ...

class BridgeBiddingEnvWrapper:
    def __init__(
        self, deal_manager: BridgeDealManager, greedy: List[int], replay: Replay
    ): ...

class BridgeWrapperVecEnv:
    def __init__(self): ...
    def push(self, env: BridgeBiddingEnvWrapper): ...

class BridgeVecEnv:
    def __init__(self):
        """
        A vectorized env to run bridge games in parallel.
        """
        ...
    def push(self, env: BridgeBiddingEnv):
        """
        Append a BridgeBiddingEnv to the vectorized env.
        Args:
            env: The env to be pushed

        Returns:
            No returns.
        """
        ...
    def size(self) -> int:
        """
        Get the size of the vectorized env. i.e. how many envs are in the vector.
        Returns:
            The number of envs.
        """
        ...
    def reset(self, obs: TensorDict) -> TensorDict:
        """
        Reset the vectorized env and get initial obs.
        Returns:
            The initial obs tensor, shape (num_envs, obs_size)
        """
        ...
    def step(self, action: TensorDict) -> Tuple[TensorDict, torch.Tensor, torch.Tensor]:
        """
        Step a batch of actions to envs, and get a batch of next obs, reward and terminal.
        If an env is terminated, the next obs tensor is fake.
        Args:
            action: The action tensor.

        Returns:
            The next obs tensor, reward tensor and terminal tensor.
        """
        ...
    def all_terminated(self) -> bool:
        """
        Whether the envs are all terminated.
        Returns:
            A boolean value indicates whether all terminated.
        """
        ...
    def get_envs(self) -> List[BridgeBiddingEnv]: ...
    def get_returns(self, player: Player) -> List[int]: ...

class ImpEnv:
    def __init__(self, deal_manager: BridgeDealManager, greedy: List[int]):
        """
        An imp env which a deal is played twice. The player sit at NS first time should play at EW for second time.
        Args:
            deal_manager: The deal manager stores deals.
            greedy: The greedy list.
        """
        ...
    def get_acting_player(self) -> int:
        """
        Get the player index. If it's the first game, the index is same as seat.
        For the second game, the player moved clockwise, i.e. index 0 plays ar East.
        Returns:
            The acting player index.
        """
        ...
    def returns(self) -> List[int]:
        """
        Get imps for 2 teams.
        Returns:
            The imp list.
        """
        ...
    def step(self, action: TensorDict) -> Tuple[TensorDict, float, bool]:
        """
        Make a step in env.
        Args:
            action: The action to step.

        Returns:
            Next obs tensor, reward and terminal.
        """
        ...
    def reset(self) -> TensorDict:
        """
        Reset the env and get initial obs.
        Returns:
            The initial obs tensor.
        """
        ...
    def terminated(self) -> bool:
        """
        Whether the env is terminated.
        Returns:
            A boolean value of terminal.
        """
        ...
    def num_states(self) -> int:
        """
        Get the number of deals played in the env.
        Returns:
            The number of deals.
        """
        ...
    def __repr__(self) -> str: ...

class Replay:
    def __init__(
        self, capacity: int, seed: int, alpha: float, beta: float, prefetch: int
    ): ...
    def sample(
        self, batch_size: int, device: str
    ) -> Tuple[Transition, torch.Tensor]: ...
    def update_priority(self, priority: torch.Tensor): ...
    def num_add(self) -> int: ...
    def size(self) -> int: ...

class ImpEnvWrapper:
    def __init__(
        self, deal_manager: BridgeDealManager, greedy: List[int], replay_buffer: Replay
    ): ...
    def get_feature(self) -> TensorDict: ...
    def step(self, reply: TensorDict): ...
    def terminated(self) -> bool: ...
    def reset(self): ...
    def __repr__(self): ...

class ImpVecEnv:
    def __init__(self): ...
    def push(self, env: ImpEnvWrapper): ...
    def reset(self): ...
    def step(self, reply: TensorDict): ...
    def get_feature(self) -> TensorDict: ...
    def get_envs(self) -> List[ImpEnvWrapper]: ...
    def any_terminated(self) -> bool: ...
    def all_terminated(self) -> bool: ...

class ThreadLoop: ...

class VecEnvEvalThreadLoop(ThreadLoop):
    def __init__(
        self,
        train_actor: VecEnvActor,
        oppo_actor: VecEnvActor,
        vec_env_0: BridgeVecEnv,
        vec_env_1: BridgeVecEnv,
    ):
        """
        A thread loop evaluates between 2 actors using vectorized env.(Recommended)
        Args:
            vec_env_0: The vec env to evaluate.
            vec_env_1: The vec env as second game, should contain same env as first one.
            train_actor: The trained actor.
            oppo_actor: The opponent actor.
        """
        ...
    def main_loop(self): ...

class ImpThreadLoop(ThreadLoop):
    def __init__(self, env: ImpVecEnv, actor: VecEnvActor): ...

class BridgeVecEnvThreadLoop(ThreadLoop):
    def __init__(self, env: BridgeWrapperVecEnv, actor: VecEnvActor): ...

class Context:
    def __init__(self):
        """
        A context to run thread loop in parallel.
        """
        ...
    def push_thread_loop(self, thread_loop: ThreadLoop):
        """
        Push a thread loop into context.
        Args:
            thread_loop: The thread loop to be pushed.

        Returns:
            No returns.
        """
        ...
    def start(self):
        """
        Start all the thread loops
        Returns:
            No returns.
        """
        ...
    def pause(self):
        """
        Pause all the thread loops.
        Returns:
            No returns.
        """
        ...
    def all_paused(self) -> bool:
        """
        Whether all the thread loops are paused.
        Returns:
            A boolean value.
        """
        ...
    def resume(self):
        """
        Resume all the thread loops.
        Returns:
            No returns.
        """
        ...
    def terminate(self):
        """
        Terminate all the thread loops.
        Returns:
            No returns.
        """
        ...
    def terminated(self) -> bool:
        """
        Whether all the thread loops are terminated.
        Returns:
            A boolean value.
        """
        ...

def get_imp(my: int, other: int) -> int:
    """
    Get imps of given duplicate scores.
    Args:
        my: The score of the team played at NS in first game.
        other: The other team's score.

    Returns:
        int: The imp.
    """
    ...

def bid_str_to_action(bid_str: str) -> Action:
    """
    Convert a bid str to action, bluechip
    Args:
        bid_str(str): The bid string, only for contract bids, i.e. not contain pass, double, redouble

    Returns:
        int: The action

    """
    ...

def bid_action_to_str(bid: int) -> str:
    """
    Convert a bid action to string, bluechip
    Args:
        bid(int): The bid action

    Returns:
        str: The string uses for bluechip protocol
    """
    ...

def get_hand_string(cards: CardsLike) -> str:
    """
    Get the string of hand according to bluechip protocol
    Args:
        cards: The list or ndarray of cards

    Returns:
        str: The string of hand
    """

def check_prob_not_zero(action: torch.Tensor, log_probs: torch.Tensor): ...
def make_obs_tensor_dict(state: BridgeBiddingState, greedy: int) -> TensorDict:
    """
    Get a tensor dict as obs for current state.
    Args:
        state: The BridgeBiddingState instance.
        greedy: 1 for greedy, 0 for stochastic.

    Returns:
        TensorDict: The obs.
    """
    ...

class Transition:
    obs: TensorDict = ...
    reply: TensorDict = ...
    reward: torch.Tensor = ...
    terminal: torch.Tensor = ...
    next_obs: TensorDict = ...

    def __init__(
        self,
        obs: TensorDict,
        reply: TensorDict,
        reward: torch.Tensor,
        terminal: torch.Tensor,
        next_obs: TensorDict,
    ): ...
    def to_dict(self) -> TensorDict:
        """
        Convert transition to tensor dict
        Returns:
            The dict.
        """
        ...

def bid_string(action: Action) -> str: ...

class VecEnvAllTerminateThreadLoop(ThreadLoop):
    def __init__(self, actor: VecEnvActor, env: BridgeVecEnv): ...

class ObsBelief:
    obs: TensorDict
    belief: TensorDict
    length: torch.Tensor

class ObsBeliefReplay:
    def __init__(
        self, capacity: int, seed: int, alpha: float, beta: float, prefetch: int
    ): ...
    def sample(
        self, batch_size: int, device: str
    ) -> Tuple[ObsBelief, torch.Tensor]: ...
    def update_priority(self, priority: torch.Tensor): ...
    def num_add(self) -> int: ...
    def size(self) -> int: ...

class BeliefThreadLoop(ThreadLoop):
    def __init__(
        self, actor: VecEnvActor, env: BridgeVecEnv, replay: ObsBeliefReplay
    ): ...
    def main_loop(self): ...

class BeliefModel:
    def __init__(self, model_locker: ModelLocker): ...
    def sample(
        self, obs: TensorDict, current_player: Player, num_sample: int, max_try: int
    ): ...

class SearchParams:
    min_rollouts: int
    max_rollouts: int
    max_particles: int
    min_prob: int
    temperature: float
    top_k: int
    filter_batch_size: int
    random_sample: bool
    verbose: bool

class Searcher:
    def __init__(
        self, params: SearchParams, belief_model: BeliefModel, actor: VecEnvActor
    ): ...
    def search(
        self, state: BridgeBiddingState, obs: TensorDict, reply: TensorDict
    ) -> TensorDict: ...

class RandomDealManager:
    def __init__(self, seed: int): ...
    def next(self) -> BridgeDeal: ...

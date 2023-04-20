from typing import List, overload, Tuple, NoReturn, Union, Optional

import numpy as np
import torch

Player = int
Action = int
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


class Actor:
    ...


class RandomActor(Actor):
    def __init__(self):
        ...

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        ...


class SingleEnvActor(Actor):
    def __init__(self, model_locker: ModelLocker, player: int, gamma: float, eval_: bool):
        """
        An actor acts for a single environment
        Args:
            model_locker: The model locker saves ScriptModule to call
            player: Where the actor plays(Actually it has no use)
            gamma: The discount factor
            eval_: Whether store transitions, set to True if you don't need to store transitions.
        """
        ...

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action for given obs tensor
        Args:
            obs: The obs tensor

        Returns:
            The action
        """
        ...


class VecEnvActor(Actor):
    def __init__(self, model_locker: ModelLocker):
        """
        An actor acts for vectorized environment
        Args:
            model_locker: The model locker saves ScriptModule to call
        """
        ...

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action for given obs tensor
        Args:
            obs: The obs tensor

        Returns:
            The action
        """
        ...


class ReplayBuffer:
    def __init__(self, state_size: int, num_actions: int, capacity: int):
        """
        A buffer stores states(obs), actions, rewards and log_probs
        Args:
            state_size: The size of state tensor
            num_actions: The number of actions
            capacity: The capacity of the buffer
        """
        ...

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


class BridgeDeal:
    def __init__(self):
        self.cards: CardsLike = ...
        self.dealer: Player = ...
        self.is_dealer_vulnerable: bool = ...
        self.is_non_dealer_vulnerable: bool = ...
        self.ddt: Optional[DDTLike] = ...
        self.par_score: Optional[int] = ...


class BridgeDealManager:
    def __init__(self, cards: CardsLike, ddts: DDTLike, par_scores: Union[List, np.ndarray]):
        ...

    def size(self) -> int:
        ...


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

    @overload
    def observation_tensor(self) -> List[float]:
        """
        Get the observation tensor
        Returns:
            The obs tensor.
        """
        ...

    @overload
    def observation_tensor(self, player: Player) -> List[float]:
        ...

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

    def __repr__(self) -> str:
        ...


def make_obs_tensor(state: BridgeBiddingState, greedy: int) -> torch.Tensor:
    """
    Get a obs tensor with given state and greedy. The obs tensor contains state obs, legal actions and whether greedy.
    Args:
        state: The BridgeBiddingState
        greedy: whether the actor should choose greedy action.

    Returns:
        The obs tensor, shape 519(480+38+1)
    """
    ...


class BridgeBiddingEnv:
    def __init__(self, deal_manager: BridgeDealManager, greedy: List[int]):
        """
        A Bridge Bidding Environment.
        Args:
            deal_manager: The deal manager stores deals.
            greedy: Whether greedy for four players. e.g. [1,1,1,1] means all the players are greedy.
        """
        ...

    def reset(self) -> torch.Tensor:
        """
        Reset the state and return a initial obs. Should be called at first of each game.
        Returns:
            The initial obs tensor.
        """
        ...

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Make one step in environment and get next obs, reward and terminal signal.
        Args:
            action: The action to step.

        Returns:
            next obs tensor, reward and terminal.
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

    def num_states(self) -> int:
        """
        Get the number of deals played in the env.
        Returns:
            The number of deals.
        """
        ...

    def __repr__(self) -> str:
        ...


class BridgeBiddingEnv2:
    def __init__(self, deal_manager: BridgeDealManager, greedy: List[int]):
        """
        A Bridge Bidding Environment using real score - par score as reward.
        Args:
            deal_manager: The deal manager stores deals.
            greedy: Whether greedy for four players. e.g. [1,1,1,1] means all the players are greedy.
        """
        ...

    def reset(self) -> torch.Tensor:
        """
        Reset the state and return a initial obs. Should be called at first of each game.
        Returns:
            The initial obs tensor.
        """
        ...

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Make one step in environment and get next obs, reward and terminal signal.
        Args:
            action: The action to step.

        Returns:
            next obs tensor, reward and terminal.
        """
        ...

    def returns(self) -> List[float]:
        """
        Get rewards as real score - par score for each player. Should be called after the env terminated.
        Returns:
            The list of rewards.
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

    def num_states(self) -> int:
        """
        Get the number of deals played in the env.
        Returns:
            The number of deals.
        """
        ...

    def __repr__(self) -> str:
        ...


class BridgeVecEnv:
    def __init__(self):
        """
        A vectorized env to run bridge games in parallel. Only used for faster evaluation.
        """
        ...

    def append(self, env: BridgeBiddingEnv):
        """
        Append a BridgeBiddingEnv to the vectorized env.
        Args:
            env: The env to be apped

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

    def reset(self) -> torch.Tensor:
        """
        Reset the vectorized env and get initial obs.
        Returns:
            The initial obs tensor, shape (num_envs, obs_size)
        """
        ...

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step a batch of actions to envs, and get a batch of next obs, reward and terminal.
        If a env is terminated, the next obs tensor is fake.
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

    def returns(self, player: Player) -> List[float]:
        """
        Get returns for a specific player of every env.
        Args:
            player: The player's id

        Returns:
            The list of returns.
        """
        ...

    def display(self, num_envs: int):
        """
        Print some envs.
        Args:
            num_envs: How many envs to print.
        Returns:
            No returns.
        """
        ...


class ImpEnv:
    def __init__(self, deal_manager:BridgeDealManager, greedy: List[int]):
        """
        An imp env which a deal is played twice. The player sit at NS first time should play at EW for second time.
        Args:
            deal_manager: The deal manager stores deals.
            greedy: The greedy list.
        """
        ...

    def acting_player(self) -> int:
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

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Make a step in env.
        Args:
            action: The action to step.

        Returns:
            Next obs tensor, reward and terminal.
        """
        ...

    def reset(self) -> torch.Tensor:
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

    def __repr__(self) -> str:
        ...


class IntConVec:
    def __init__(self):
        """
        A concurrent vector saves integer items.
        """
        ...

    def push_back(self, item: int):
        """
        Push an item to the vector
        Args:
            item: the item to store

        Returns:
            No returns.
        """
        ...

    def empty(self) -> bool:
        """
        Whether the vector is empty.
        Returns:
            A boolean value.
        """
        ...

    def size(self) -> int:
        """
        Get the size of vector.
        Returns:
            An integer of size.
        """
        ...

    def get_vector(self) -> List[int]:
        """
        Get the vector.
        Returns:
            A list stores integers.
        """
        ...

    def clear(self):
        """
        Clear the vector.
        Returns:
            No returns.
        """


class ThreadLoop:
    ...


class BridgePGThreadLoop(ThreadLoop):
    def __init__(self,
                 actors: List[SingleEnvActor],
                 env_0: BridgeBiddingEnv,
                 env_1: BridgeBiddingEnv,
                 buffer: ReplayBuffer,
                 verbose: bool):
        """
        A thread loop to play games between actors infinitely.
        Args:
            actors: A list of actors. actors[0] and actors[2] should be trained actors.
            env_0: The env to play.
            env_1: The env to play, should contain same deals with env_0
            buffer: The replay buffer.
            verbose: Whether to show some messages.
        """
        ...

    def main_loop(self):
        ...


class BridgeThreadLoop(ThreadLoop):
    def __init__(self, actors: List[SingleEnvActor], env: BridgeBiddingEnv2, buffer: ReplayBuffer, verbose: bool):
        ...


class ImpEnvThreadLoop(ThreadLoop):
    def __init__(self, actors: List[SingleEnvActor], imp_env: ImpEnv, buffer: ReplayBuffer, verbose: bool):
        """
        A thread loop plays between actors infinitely. Should be used with Context.
        Args:
            actors: A list of actors. actors[0] and actors[2] should be trained actors.
            imp_env: The imp env.
            buffer: The replay buffer.
            verbose: Whether to print some message.
        """
        ...

    def main_loop(self):
        ...


class EvalThreadLoop(ThreadLoop):
    def __init__(self,
                 env_0: BridgeBiddingEnv,
                 env_1: BridgeBiddingEnv,
                 actor_train: SingleEnvActor,
                 actor_oppo: SingleEnvActor,
                 num_deals: int,
                 imp_vec: IntConVec,
                 verbose: bool):
        """
        A thread loop evaluates between 2 actors using single env.
        Args:
            env_0: The env to evaluate.
            env_1: The env to evaluate, should store same deals as env_1.
            actor_train: The trained actor.
            actor_oppo: The opponent actor.
            num_deals: The number of deals to play.
            imp_vec: The concurrent vector stores imp.
            verbose: Whether print some messages.

        Examples:
            t = EvalThreadLoop(env_0, env_1, actor_train, actor_oppo, num_deals, imp_vec, verbose)
            t.main_loop()
            imps = imp_vec.get_vector()
        """
        ...

    def main_loop(self):
        ...


class VecEvalThreadLoop(ThreadLoop):
    def __init__(self,
                 vec_env_0: BridgeVecEnv,
                 vec_env_1: BridgeVecEnv,
                 actor_train: VecEnvActor,
                 actor_oppo: VecEnvActor,
                 imp_vec: IntConVec,
                 verbose: bool,
                 num_loops: int):
        """
        A thread loop evaluates between 2 actors using vectorized env.(Recommended)
        Args:
            vec_env_0: The vec env to evaluate.
            vec_env_1: The vec env as second game, should contain same env as first one.
            actor_train: The trained actor.
            actor_oppo: The opponent actor.
            imp_vec: The concurrent vector stores imp.
            verbose: Whether to print some messages.
            num_loops: How many loops to play.
        """
        ...

    def main_loop(self):
        ...


class EvalImpThreadLoop(ThreadLoop):
    def __init__(self, actors: List[SingleEnvActor], env: ImpEnv, num_deals: int):
        """
        A thread loop evaluates using imp env. (Not recommended.)
        Args:
            actors: A list of actors. actors[0] plays at North in first game and East in second game.
            env: The imp env.
            num_deals: Number of deals to be played.
        """
        ...

    def main_loop(self):
        ...


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


def calc_all_tables(cards: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    ...


def generate_deals(num_deals: int, seed: int) -> Tuple[List[List[Action]], List[List[int]], List[int]]:
    ...

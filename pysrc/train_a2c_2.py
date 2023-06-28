import copy
import os
import pprint
import time
from typing import Dict, Optional

import torch
from tqdm import trange
from adan import Adan

import rl_cpp
import common_utils
import yaml

from bridge_consts import NUM_CALLS, PLUS_MINUS_SYMBOL
from utils import sl_net, load_rl_dataset, Evaluator
from agent_for_cpp import VecEnvAgent


def main():
    torch.set_printoptions(threshold=100000)
    with open("conf/a2c_2.yaml") as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    # seed and utils
    save_dir = common_utils.mkdir_with_increment(cfg["save_dir"])
    top_k = cfg["top_k"]
    common_utils.set_random_seeds(cfg["seed"])
    stats = common_utils.MultiStats()
    logger = common_utils.Logger(os.path.join(save_dir, "log.txt"), verbose=True, auto_line_feed=True)
    logger.write(pprint.pformat(cfg))
    saver = common_utils.TopKSaver(save_dir, top_k)
    stopwatch = common_utils.Stopwatch()
    net = sl_net()
    supervised_net = sl_net()
    p_lr = cfg["policy_lr"]
    v_lr = cfg["value_lr"]
    train_device = cfg["train_device"]
    agent = VecEnvAgent(net).to(train_device)
    agent.v_net.load_state_dict(torch.load("models/value.pth"))
    act_device = cfg["act_device"]

    train_locker = rl_cpp.ModelLocker([torch.jit.script(copy.deepcopy(agent)).to(act_device)], act_device)
    train_actor = rl_cpp.VecEnvActor(train_locker)
    # p_opt = torch.optim.AdamW(params=agent.p_net.parameters(), lr=p_lr)
    # v_opt = torch.optim.AdamW(params=agent.v_net.parameters(), lr=v_lr)
    p_opt = Adan(params=agent.p_net.parameters(), lr=p_lr, fused=True)
    v_opt = Adan(params=agent.v_net.parameters(), lr=v_lr, fused=True)

    # p_lr_scheduler = torch.optim.lr_scheduler.StepLR(p_opt, step_size=200 * 1000, gamma=0.1)
    # v_lr_scheduler = torch.optim.lr_scheduler.StepLR(v_opt, step_size=200 * 1000, gamma=0.1)

    # evaluator
    num_eval_deals = cfg["num_eval_deals"]
    num_eval_threads = cfg["num_eval_threads"]
    eval_device = cfg["eval_device"]
    evaluator = Evaluator(num_eval_deals, num_eval_threads, eval_device)

    # envs
    num_threads = cfg["num_threads"]
    num_games_per_thread = cfg["num_games_per_thread"]
    dataset = load_rl_dataset("train3")
    deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    buffer_capacity = cfg["buffer_capacity"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    prefetch = cfg["prefetch"]
    replay_buffer = rl_cpp.Replay(buffer_capacity, cfg["seed"], alpha, beta, prefetch)
    context = rl_cpp.Context()
    for _ in trange(num_threads):
        vec_env = rl_cpp.BridgeWrapperVecEnv()
        for i_env in range(num_games_per_thread):
            env = rl_cpp.BridgeBiddingEnvWrapper(deal_manager, [0, 0, 0, 0], replay_buffer)
            vec_env.push(env)
        t = rl_cpp.BridgeVecEnvThreadLoop(vec_env, train_actor)
        context.push_thread_loop(t)

    max_grad_norm = cfg["max_grad_norm"]
    burn_in_frames = cfg["burn_in_frames"]
    context.start()
    while (size := replay_buffer.size()) < burn_in_frames:
        print(f"\rWarming up replay buffer, {size} / {burn_in_frames}", end="")
        time.sleep(1)
    print()

    # train/eval loop
    num_epochs = cfg["num_epochs"]
    epoch_len = cfg["epoch_len"]
    sample_batch_size = cfg["sample_batch_size"]
    entropy_ratio = cfg["entropy_ratio"]
    clip_eps = cfg["clip_eps"]
    sync_freq = cfg["sync_freq"]

    for i_ep in range(num_epochs):
        st = time.perf_counter()
        mem_usage = common_utils.get_mem_usage()
        print(f"Beginning of Epoch {i_ep}\nMem usage: {mem_usage}")
        stopwatch.reset()

        for batch_idx in trange(epoch_len):
            num_update = batch_idx + i_ep * epoch_len
            torch.cuda.synchronize()
            p_opt.zero_grad()
            v_opt.zero_grad()

            if num_update % sync_freq == 0:
                train_locker.update_model(torch.jit.script(copy.deepcopy(agent)).to(act_device))

            stopwatch.time("sync and updating")

            batch, weights = replay_buffer.sample(sample_batch_size, train_device)
            # print(weights)
            stopwatch.time("sample data")

            p_loss, v_loss, priority = agent.compute_loss_and_priority(batch, clip_eps, entropy_ratio)
            # print(p_loss, v_loss)
            # print(v_loss.mean())
            torch.cuda.synchronize()

            # stats.feed("p_loss", p_loss.item())
            stats.feed("v_loss", v_loss.mean().item())

            weighted_loss = ((p_loss + v_loss) * weights).mean()
            torch.cuda.synchronize()
            stats.feed("weighted_loss", weighted_loss.item())
            stopwatch.time("calculating loss")
            # print(weighted_loss)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.p_net.parameters(), max_grad_norm)
            p_opt.step()
            v_opt.step()
            # p_lr_scheduler.step()
            # v_lr_scheduler.step()
            torch.cuda.synchronize()
            stopwatch.time("backprop & update")

            replay_buffer.update_priority(priority)

            stopwatch.time("update priority")

            # stats.feed("num_add", replay_buffer.num_add())

        stopwatch.summary()
        context.pause()
        while not context.all_paused():
            time.sleep(0.5)

        avg, sem, elapsed_time, _, _ = evaluator.evaluate(agent.p_net, supervised_net)
        stats.feed("avg_imp", avg)
        stats.feed("sem_imp", sem)
        msg = f"Epoch {i_ep}, result: {avg}{PLUS_MINUS_SYMBOL}{sem:.4f}, elapsed time: {elapsed_time:.2f}."
        logger.write(msg)
        checkpoint = {
            "model_state_dict": {
                "policy": copy.deepcopy(agent.p_net.state_dict()),
                "value": copy.deepcopy(agent.v_net.state_dict())
            }
        }
        saver.save(checkpoint, avg, True)
        stats.save_all(save_dir, True)
        ed = time.perf_counter()
        print(f"Epoch {i_ep}, total time: {ed - st:.2f}")
        context.resume()

    context.terminate()


if __name__ == '__main__':
    main()

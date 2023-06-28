import argparse
import copy
import os
import time

import numpy as np
import torch
import yaml
from adan import Adan

import rl_cpp
import common_utils
from agent_for_cpp import SingleEnvAgent, VecEnvAgent
from nets import PolicyNet
from utils import load_rl_dataset, sl_net, Evaluator

if __name__ == '__main__':
    with open("conf/search.yaml") as fp:
        cfg = yaml.safe_load(fp)

    dataset = load_rl_dataset("train")
    deal_manager = rl_cpp.BridgeDealManager(dataset["cards"], dataset["ddts"], dataset["par_scores"])
    net = sl_net()
    supervised_net = sl_net()
    s_agent = SingleEnvAgent(net)
    s_agent.perfect_v_net.load_state_dict(torch.load("models/value.pth"))
    agent = VecEnvAgent(net)
    agent.v_net.load_state_dict(torch.load("models/value.pth"))
    act_device = cfg["act_device"]
    s_agent = s_agent.to(act_device)
    agent = agent.to(act_device)
    p_opt = Adan(params=agent.p_net.parameters(), lr=cfg["policy_lr"], fused=True)
    v_opt = Adan(params=agent.v_net.parameters(), lr=cfg["value_lr"], fused=True)

    single_locker = rl_cpp.ModelLocker([torch.jit.script(copy.deepcopy(s_agent)).to(act_device)], act_device)
    s_actor = rl_cpp.SingleEnvActor(single_locker)
    vec_locker = rl_cpp.ModelLocker([torch.jit.script(copy.deepcopy(agent)).to(act_device)], act_device)
    actor = rl_cpp.VecEnvActor(vec_locker)

    params = rl_cpp.SearchParams()
    params.top_k = cfg["k"]
    params.min_prob = cfg["min_prob"]
    params.temperature = cfg["temperature"]
    params.max_rollouts = cfg["num_rollouts"]
    params.action_selection = 0

    actors = [actor for _ in range(4)]
    evaluator = Evaluator(cfg["num_eval_deals"], cfg["num_eval_threads"], cfg["eval_device"])
    searcher = rl_cpp.PerfectSearcher(params, actors)
    replay = rl_cpp.PVReplay(cfg["buffer_capacity"], 0, 0, 0.5, 0)
    context = rl_cpp.Context()

    for i_t in range(cfg["num_threads"]):
        t = rl_cpp.SearchThreadLoop2(s_actor, deal_manager, searcher, replay)
        context.push_thread_loop(t)

    context.start()

    num_update_transitions = 10000
    num_transitions = num_update_transitions

    print("start.")

    while True:
        if replay.num_add() >= num_transitions:
            num_transitions += num_update_transitions
            batch, w = replay.sample(num_update_transitions, cfg["train_device"])
            for i in range(cfg["num_gd_steps"]):
                p_opt.zero_grad()
                policy_loss = agent.search_loss(batch)
                print(policy_loss.mean())
                policy_loss.mean().backward()
                p_opt.step()

            replay.update_priority(torch.tensor([]))
            print("priority updated.")
            context.pause()
            while not context.all_paused():
                time.sleep(0.5)
            print("start evaluating.")
            avg, sem, *_ = evaluator.evaluate(agent.p_net, supervised_net)
            print(avg, sem)
            s_agent.p_net.load_state_dict(agent.p_net.state_dict())
            vec_locker.update_model(torch.jit.script(copy.deepcopy(agent).to(act_device)))
            single_locker.update_model(torch.jit.script(copy.deepcopy(s_agent).to(act_device)))
            context.resume()
        else:
            print(f"\r{replay.num_add()}", end="")
            time.sleep(20)

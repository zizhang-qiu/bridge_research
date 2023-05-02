//
// Created by qzz on 2023/2/20.
//
#include "base.h"
#include "bluechip_utils.h"
#include "bridge_actor.h"
#include "bridge_envs.h"
#include "bridge_scoring.h"
#include "multi_agent_transition_buffer.h"
#include "bridge_state.h"
#include "context.h"
#include "encode_bridge.h"
#include "generate_deals.h"
#include "model_locker.h"
#include "replay_buffer.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "thread_loop.h"
#include "imp_env.h"
#include "search.h"

namespace py = pybind11;
using namespace bridge_encode;
using namespace rl;
using namespace rl::bridge;

PYBIND11_MODULE(rl_cpp, m) {
  m.def("encode", &Encode, "A function encodes bridge state.");
  m.def("get_imp", &GetImp);

  py::class_<rl::ModelLocker, std::shared_ptr<rl::ModelLocker>>(m,
                                                                "ModelLocker")
      .def(py::init<std::vector<py::object>, const std::string>())
      .def("update_model", &ModelLocker::UpdateModel);

  py::class_<RandomActor, std::shared_ptr<RandomActor>>(m, "RandomActor")
      .def(py::init<>())
      .def("act", &RandomActor::Act);

  py::class_<SingleEnvActor, std::shared_ptr<SingleEnvActor>>(
      m, "SingleEnvActor")
      .def(py::init<std::shared_ptr<ModelLocker>>())
      .def("act", &SingleEnvActor::Act)
      .def("get_top_k_actions_with_min_prob", &SingleEnvActor::GetTopKActionsWithMinProb)
      .def("get_prob_for_action", &SingleEnvActor::GetProbForAction);

  py::class_<VecEnvActor, std::shared_ptr<VecEnvActor>>(m, "VecEnvActor")
      .def(py::init<std::shared_ptr<ModelLocker>>())
      .def("act", &VecEnvActor::Act);

  py::class_<ReplayBuffer, std::shared_ptr<ReplayBuffer>>(m, "ReplayBuffer")
      .def(py::init<int, int, int>())
      .def("push", &ReplayBuffer::Push)
      .def("sample", &ReplayBuffer::Sample)
      .def("size", &ReplayBuffer::Size)
      .def("get_all", &ReplayBuffer::GetAll)
      .def("num_add", &ReplayBuffer::NumAdd)
      .def("dump", &ReplayBuffer::Dump)
      .def("load", &ReplayBuffer::Load);

  py::class_<BridgeDeal, std::shared_ptr<BridgeDeal>>(m, "BridgeDeal")
      .def(py::init<>())
      .def_readwrite("cards", &BridgeDeal::cards)
      .def_readwrite("ddt", &BridgeDeal::ddt)
      .def_readwrite("par_score", &BridgeDeal::par_score)
      .def_readwrite("dealer", &BridgeDeal::dealer)
      .def_readwrite("is_dealer_vulnerable", &BridgeDeal::is_dealer_vulnerable)
      .def_readwrite("is_non_dealer_vulnerable", &BridgeDeal::is_non_dealer_vulnerable);

  py::class_<Contract>(m, "Contract")
      .def_readonly("level", &Contract::level)
      .def("trumps", [](const Contract& c){return static_cast<int>(c.trumps);})
      .def("double_status", [](const Contract& c){return static_cast<int>(c.double_status);})
      .def_readonly("declarer", &Contract::declarer);

  py::class_<BridgeBiddingState, std::shared_ptr<BridgeBiddingState>>(
      m, "BridgeBiddingState")
      .def(py::init<BridgeDeal>())
//      .def(py::init<Player,
//                    std::vector<Action>, // cards
//                    bool,                // is dealer vulnerable
//                    bool,                // is non-dealer vulnerable
//                    const std::vector<int>>())
      .def("history", &BridgeBiddingState::History)
      .def("bid_str", &BridgeBiddingState::BidStr)
      .def("bid_history", &BridgeBiddingState::BidHistory)
      .def("bid_str_history", &BridgeBiddingState::BidStrHistory)
      .def("apply_action", &BridgeBiddingState::ApplyAction)
      .def("terminated", &BridgeBiddingState::Terminated)
      .def("returns", &BridgeBiddingState::Returns)
      .def("contract_str", &BridgeBiddingState::ContractString)
      .def("observation_str", &BridgeBiddingState::ObservationString)
      .def("current_player", &BridgeBiddingState::CurrentPlayer)
      .def("current_phase", &BridgeBiddingState::CurrentPhase)
      .def("get_actual_trick_and_dd_trick", &BridgeBiddingState::GetActualTrickAndDDTrick)
      .def("get_contract", &BridgeBiddingState::GetContract)
      .def("observation_tensor",
           py::overload_cast<Player>(&BridgeBiddingState::ObservationTensor))
      .def("observation_tensor",
           py::overload_cast<>(&BridgeBiddingState::ObservationTensor))
      .def("terminate", &BridgeBiddingState::Terminate)
      .def("legal_actions", &BridgeBiddingState::LegalActions)
      .def("clone", &BridgeBiddingState::Clone)
      .def("get_double_dummy_table", &BridgeBiddingState::GetDoubleDummyTable)
      .def("__repr__", &BridgeBiddingState::ToString);

  py::class_<BridgeTransitionBuffer, std::shared_ptr<BridgeTransitionBuffer>>(m, "BridgeTransitionBuffer")
      .def(py::init<>())
      .def("push_obs_action_log_probs", &BridgeTransitionBuffer::PushObsActionLogProbs)
      .def("push_to_replay_buffer", &BridgeTransitionBuffer::PushToReplayBuffer);

  py::class_<MultiAgentTransitionBuffer, std::shared_ptr<MultiAgentTransitionBuffer>>(m, "MultiAgentTransitionBuffer")
      .def(py::init<int>())
      .def("push_obs_action_log_probs", &MultiAgentTransitionBuffer::PushObsActionLogProbs)
      .def("push_to_replay_buffer", &MultiAgentTransitionBuffer::PushToReplayBuffer);

  py::class_<BridgeDealManager, std::shared_ptr<BridgeDealManager>>(m, "BridgeDealManager")
      .def(py::init<const std::vector<Cards>,
                    const std::vector<DDT>,
                    const std::vector<int>>())
      .def("size", &BridgeDealManager::Size)
      .def("next", &BridgeDealManager::Next);

  py::class_<BridgeBiddingEnv, std::shared_ptr<BridgeBiddingEnv>>(m, "BridgeBiddingEnv")
      .def(py::init<std::shared_ptr<BridgeDealManager>,
                    std::vector<int>,
                    std::shared_ptr<ReplayBuffer>,
                    bool,
                    bool
      >())
      .def("reset", &BridgeBiddingEnv::Reset)
      .def("step", &BridgeBiddingEnv::Step)
      .def("returns", &BridgeBiddingEnv::Returns)
      .def("terminated", &BridgeBiddingEnv::Terminated)
      .def("get_current_player", &BridgeBiddingEnv::GetCurrentPlayer)
      .def("get_state", &BridgeBiddingEnv::GetState)
      .def("get_num_deals", &BridgeBiddingEnv::GetNumDeals)
      .def("get_feature_size", &BridgeBiddingEnv::GetFeatureSize)
      .def("__repr__", &BridgeBiddingEnv::ToString);

  py::class_<BridgeVecEnv, std::shared_ptr<BridgeVecEnv>>(m, "BridgeVecEnv")
      .def(py::init<>())
      .def("push", &BridgeVecEnv::Push, py::keep_alive<1, 2>())
      .def("size", &BridgeVecEnv::Size)
      .def("reset", &BridgeVecEnv::Reset)
      .def("step", &BridgeVecEnv::Step)
      .def("get_envs", &BridgeVecEnv::GetEnvs)
      .def("get_returns", &BridgeVecEnv::GetReturns)
      .def("any_terminated", &BridgeVecEnv::AnyTerminated)
      .def("all_terminated", &BridgeVecEnv::AllTerminated);

  py::class_<ImpEnv, std::shared_ptr<ImpEnv>>(m, "ImpEnv")
      .def(py::init<std::shared_ptr<BridgeDealManager>,
                    std::vector<int>,
                    std::shared_ptr<ReplayBuffer>,
                    bool>())
      .def("reset", &ImpEnv::Reset)
      .def("step", &ImpEnv::Step)
      .def("terminated", &ImpEnv::Terminated)
      .def("returns", &ImpEnv::Returns)
      .def("get_acting_player", &ImpEnv::GetActingPlayer)
      .def("get_num_deals", &ImpEnv::GetNumDeals)
      .def("__repr__", &ImpEnv::ToString);

  py::class_<ImpVecEnv, std::shared_ptr<ImpVecEnv>>(m, "ImpVecEnv")
      .def(py::init<>())
      .def("push", &ImpVecEnv::Push, py::keep_alive<1, 2>())
      .def("size", &ImpVecEnv::Size)
      .def("reset", &ImpVecEnv::Reset)
      .def("step", &ImpVecEnv::Step)
      .def("any_terminated", &ImpVecEnv::AnyTerminated);

  py::class_<Context, std::shared_ptr<Context>>(m, "Context")
      .def(py::init<>())
      .def("push_thread_loop", &Context::PushThreadLoop, py::keep_alive<1, 2>())
      .def("start", &Context::Start)
      .def("pause", &Context::Pause)
      .def("all_paused", &Context::AllPaused)
      .def("resume", &Context::Resume)
      .def("terminate", &Context::Terminate)
      .def("terminated", &Context::Terminated);

  py::class_<IntConVec, std::shared_ptr<IntConVec>>(m, "IntConVec")
      .def(py::init<>())
      .def("push_back", &IntConVec::PushBack)
      .def("push_back_no_wait", &IntConVec::PushBackNoWait)
      .def("empty", &IntConVec::Empty)
      .def("size", &IntConVec::Size)
      .def("get_vector", &IntConVec::GetVector)
      .def("clear", &IntConVec::Clear);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

//  py::class_<EvalThreadLoop, ThreadLoop, std::shared_ptr<EvalThreadLoop>>(
//      m, "EvalThreadLoop")
//      .def(py::init<std::shared_ptr<bridge::BridgeBiddingEnv>,
//                    std::shared_ptr<bridge::BridgeBiddingEnv>,
//                    std::shared_ptr<bridge::SingleEnvActor>,
//                    std::shared_ptr<bridge::SingleEnvActor>, int,
//                    std::shared_ptr<IntConVec>, bool>())
//      .def("main_loop", &EvalThreadLoop::MainLoop);

  py::class_<VecEnvEvalThreadLoop, ThreadLoop, std::shared_ptr<VecEnvEvalThreadLoop>>(m, "VecEnvEvalThreadLoop")
      .def(py::init<
          std::shared_ptr<bridge::VecEnvActor>,
          std::shared_ptr<bridge::VecEnvActor>,
          std::shared_ptr<BridgeVecEnv>,
          std::shared_ptr<BridgeVecEnv>
      >())
      .def("main_loop", &VecEnvEvalThreadLoop::MainLoop);

  py::class_<bridge::BridgeThreadLoop, ThreadLoop, std::shared_ptr<BridgeThreadLoop>>(m, "BridgeThreadLoop")
      .def(py::init<std::shared_ptr<BridgeVecEnv>,
                    std::shared_ptr<VecEnvActor>>());

  py::class_<bridge::ImpThreadLoop, ThreadLoop, std::shared_ptr<ImpThreadLoop>>(m, "ImpThreadLoop")
      .def(py::init<std::shared_ptr<ImpVecEnv>,
                    std::shared_ptr<VecEnvActor>>());

  py::class_<ImpSingleEnvThreadLoop, ThreadLoop, std::shared_ptr<ImpSingleEnvThreadLoop>>(m, "ImpSingleEnvThreadLoop")
      .def(py::init<std::shared_ptr<SingleEnvActor>, std::shared_ptr<SingleEnvActor>, std::shared_ptr<ImpEnv>, bool>())
      .def("main_loop", &ImpSingleEnvThreadLoop::MainLoop);

//  py::class_<BridgeThreadLoop, ThreadLoop, std::shared_ptr<BridgeThreadLoop>>(m, "BridgeThreadLoop")
//      .def(py::init<
//          std::vector<std::shared_ptr<bridge::SingleEnvActor>>,
//          std::shared_ptr<bridge::BridgeBiddingEnv2>,
//          std::shared_ptr<bridge::ReplayBuffer>,
//          bool>())
//      .def("main_loop", &BridgeThreadLoop::MainLoop);

//  py::class_<ImpEnvThreadLoop, ThreadLoop, std::shared_ptr<ImpEnvThreadLoop>>(
//      m, "ImpEnvThreadLoop")
//      .def(py::init<std::vector<std::shared_ptr<bridge::SingleEnvActor>>,
//                    std::shared_ptr<bridge::ImpEnv>,
//                    std::shared_ptr<bridge::ReplayBuffer>, bool>())
//      .def("main_loop", &ImpEnvThreadLoop::MainLoop);
//
//  py::class_<EvalImpThreadLoop, ThreadLoop, std::shared_ptr<EvalImpThreadLoop>>(
//      m, "EvalImpThreadLoop")
//      .def(py::init<std::vector<std::shared_ptr<bridge::SingleEnvActor>>,
//                    std::shared_ptr<bridge::ImpEnv>, const int>())
//      .def("main_loop", &EvalImpThreadLoop::MainLoop);
  m.def("make_obs_tensor_dict", &bridge::MakeObsTensorDict);
  m.def("check_prob_not_zero", &rl::utils::CheckProbNotZero);

  m.def("bid_str_to_action", &bluechip::BidStrToAction);
  m.def("bid_action_to_str", &bluechip::BidActionToStr);
  m.def("get_hand_string", &bluechip::GetHandString);

  m.def("calc_all_tables", &rl::bridge::CalcAllTables);
  m.def("generate_deals", &rl::bridge::GenerateDeals);

  py::class_<SearchParams>(m, "SearchParams")
      .def(py::init<>())
      .def_readwrite("min_rollouts", &SearchParams::min_rollouts)
      .def_readwrite("max_rollouts", &SearchParams::max_rollouts)
      .def_readwrite("max_particles", &SearchParams::max_particles)
      .def_readwrite("temperature", &SearchParams::temperature)
      .def_readwrite("top_k", &SearchParams::top_k)
      .def_readwrite("min_prob", &SearchParams::min_prob)
      .def_readwrite("verbose_level", &SearchParams::verbose_level);
  m.def("search", &rl::bridge::Search);
}

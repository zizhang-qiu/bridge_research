//
// Created by qzz on 2023/2/20.
//
#include "base.h"
#include "bluechip_utils.h"
#include "bridge_actor.h"
#include "bridge_env.h"
#include "bridge_scoring.h"
#include "bridge_state.h"
#include "context.h"
#include "encode_bridge.h"
#include "generate_deals.h"
#include "model_locker.h"
#include "replay_buffer.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "thread_loop.h"

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

  py::class_<Actor, std::shared_ptr<Actor>>(m, "Actor");

  py::class_<RandomActor, Actor, std::shared_ptr<RandomActor>>(m, "RandomActor")
      .def(py::init<>())
      .def("act", &RandomActor::Act);

  py::class_<SingleEnvActor, Actor, std::shared_ptr<SingleEnvActor>>(
      m, "SingleEnvActor")
      .def(py::init<std::shared_ptr<ModelLocker>,
                    int,   // player
                    float, // gamma
                    bool>())
      .def("act", &SingleEnvActor::Act)
      .def("set_reward_and_terminal", &SingleEnvActor::SetRewardAndTerminal)
      .def("post_to_replay_buffer", &SingleEnvActor::PostToReplayBuffer);

  py::class_<VecEnvActor, Actor, std::shared_ptr<VecEnvActor>>(m, "VecEnvActor")
      .def(py::init<std::shared_ptr<ModelLocker>>())
      .def("act", &VecEnvActor::Act);

  py::class_<ReplayBuffer, std::shared_ptr<ReplayBuffer>>(m, "ReplayBuffer")
      .def(py::init<int, int, int>())
      .def("sample", &ReplayBuffer::Sample)
      .def("size", &ReplayBuffer::Size)
      .def("get_all", &ReplayBuffer::GetAll)
      .def("num_add", &ReplayBuffer::NumAdd);

  py::class_<BridgeDeal, std::shared_ptr<BridgeDeal>>(m, "BridgeDeal")
      .def(py::init<>())
      .def_readwrite("cards", &BridgeDeal::cards)
      .def_readwrite("ddt", &BridgeDeal::ddt)
      .def_readwrite("par_scores", &BridgeDeal::par_score)
      .def_readwrite("dealer", &BridgeDeal::dealer)
      .def_readwrite("is_dealer_vulnerable", &BridgeDeal::is_dealer_vulnerable)
      .def_readwrite("is_non_dealer_vulnerable", &BridgeDeal::is_non_dealer_vulnerable);

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
      .def("observation_tensor",
           py::overload_cast<Player>(&BridgeBiddingState::ObservationTensor))
      .def("observation_tensor",
           py::overload_cast<>(&BridgeBiddingState::ObservationTensor))
      .def("terminate", &BridgeBiddingState::Terminate)
      .def("legal_actions", &BridgeBiddingState::LegalActions)
      .def("clone", &BridgeBiddingState::Clone)
      .def("get_double_dummy_table", &BridgeBiddingState::GetDoubleDummyTable)
      .def("__repr__", &BridgeBiddingState::ToString);

  m.def("make_obs_tensor", &MakeObsTensor);

  py::class_<BridgeBiddingEnv, std::shared_ptr<BridgeBiddingEnv>>(
      m, "BridgeBiddingEnv")
      .def(py::init<std::vector<std::vector<Action>>,
                    std::vector<std::vector<int>>,
                    std::vector<int> // greedy
      >())
      .def(py::init<std::vector<Action>, std::vector<int>, std::vector<int>>())
      .def("reset", &BridgeBiddingEnv::Reset)
      .def("step", &BridgeBiddingEnv::Step)
      .def("returns", &BridgeBiddingEnv::Returns)
      .def("terminated", &BridgeBiddingEnv::Terminated)
      .def("get_current_cards_and_ddt",
           &BridgeBiddingEnv::GetCurrentCardsAndDDT)
      .def("make_sub_env", &BridgeBiddingEnv::MakeSubEnv)
      .def("get_state", &BridgeBiddingEnv::GetState)
      .def("num_states", &BridgeBiddingEnv::NumStates)
      .def("__repr__", &BridgeBiddingEnv::ToString);

  py::class_<BridgeBiddingEnv2, std::shared_ptr<BridgeBiddingEnv2>>(
      m, "BridgeBiddingEnv2")
      .def(py::init<std::vector<std::vector<Action>>,
                    std::vector<std::vector<int>>, std::vector<int>,
                    std::vector<int>>())
      .def("reset", &BridgeBiddingEnv2::Reset)
      .def("step", &BridgeBiddingEnv2::Step)
      .def("returns", &BridgeBiddingEnv2::Returns)
      .def("terminated", &BridgeBiddingEnv2::Terminated)
      .def("get_current_cards_and_ddt",
           &BridgeBiddingEnv2::GetCurrentCardsAndDDT)
      .def("get_state", &BridgeBiddingEnv2::GetState)
      .def("num_states", &BridgeBiddingEnv2::NumStates)
      .def("__repr__", &BridgeBiddingEnv2::ToString);

  py::class_<BridgeVecEnv, std::shared_ptr<BridgeVecEnv>>(m, "BridgeVecEnv")
      .def(py::init<>())
      .def("append", &BridgeVecEnv::Append, py::keep_alive<1, 2>())
      .def("size", &BridgeVecEnv::Size)
      .def("reset", &BridgeVecEnv::Reset)
      .def("step", &BridgeVecEnv::Step)
      .def("all_terminated", &BridgeVecEnv::AllTerminated)
      .def("returns", &BridgeVecEnv::Returns)
      .def("display", &BridgeVecEnv::Display);

  py::class_<ImpEnv, std::shared_ptr<ImpEnv>>(m, "ImpEnv")
      .def(py::init<std::vector<std::vector<int>>,
                    std::vector<std::vector<int>>, std::vector<int>, bool>())
      .def("reset", &ImpEnv::Reset)
      .def("step", &ImpEnv::Step)
      .def("terminated", &ImpEnv::Terminated)
      .def("returns", &ImpEnv::Returns)
      .def("acting_player", &ImpEnv::ActingPlayer)
      .def("num_states", &ImpEnv::NumStates)
      .def("history_imps", &ImpEnv::HistoryImps)
      .def("__repr__", &ImpEnv::ToString);

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

  py::class_<EvalThreadLoop, ThreadLoop, std::shared_ptr<EvalThreadLoop>>(
      m, "EvalThreadLoop")
      .def(py::init<std::shared_ptr<bridge::BridgeBiddingEnv>,
                    std::shared_ptr<bridge::BridgeBiddingEnv>,
                    std::shared_ptr<bridge::SingleEnvActor>,
                    std::shared_ptr<bridge::SingleEnvActor>, int,
                    std::shared_ptr<IntConVec>, bool>())
      .def("main_loop", &EvalThreadLoop::MainLoop);

  py::class_<VecEvalThreadLoop, ThreadLoop, std::shared_ptr<VecEvalThreadLoop>>(
      m, "VecEvalThreadLoop")
      .def(py::init<std::shared_ptr<bridge::BridgeVecEnv>,
                    std::shared_ptr<bridge::BridgeVecEnv>,
                    std::shared_ptr<bridge::VecEnvActor>,
                    std::shared_ptr<bridge::VecEnvActor>,
                    std::shared_ptr<IntConVec>, bool, int>())
      .def("main_loop", &VecEvalThreadLoop::MainLoop);

  py::class_<BridgePGThreadLoop, ThreadLoop,
             std::shared_ptr<BridgePGThreadLoop>>(m, "BridgePGThreadLoop")
      .def(py::init<std::vector<bridge::SingleEnvActor>,
                    std::shared_ptr<bridge::BridgeBiddingEnv>,
                    std::shared_ptr<bridge::BridgeBiddingEnv>,
                    std::shared_ptr<bridge::ReplayBuffer>, bool>())
      .def("main_loop", &BridgePGThreadLoop::MainLoop);

  py::class_<BridgeThreadLoop, ThreadLoop, std::shared_ptr<BridgeThreadLoop>>(m, "BridgeThreadLoop")
      .def(py::init<
          std::vector<std::shared_ptr<bridge::SingleEnvActor>>,
          std::shared_ptr<bridge::BridgeBiddingEnv2>,
          std::shared_ptr<bridge::ReplayBuffer>,
          bool>())
      .def("main_loop", &BridgeThreadLoop::MainLoop);

  py::class_<ImpEnvThreadLoop, ThreadLoop, std::shared_ptr<ImpEnvThreadLoop>>(
      m, "ImpEnvThreadLoop")
      .def(py::init<std::vector<std::shared_ptr<bridge::SingleEnvActor>>,
                    std::shared_ptr<bridge::ImpEnv>,
                    std::shared_ptr<bridge::ReplayBuffer>, bool>())
      .def("main_loop", &ImpEnvThreadLoop::MainLoop);

  py::class_<EvalImpThreadLoop, ThreadLoop, std::shared_ptr<EvalImpThreadLoop>>(
      m, "EvalImpThreadLoop")
      .def(py::init<std::vector<std::shared_ptr<bridge::SingleEnvActor>>,
                    std::shared_ptr<bridge::ImpEnv>, const int>())
      .def("main_loop", &EvalImpThreadLoop::MainLoop);

  m.def("bid_str_to_action", &bluechip::BidStrToAction);
  m.def("bid_action_to_str", &bluechip::BidActionToStr);
  m.def("get_hand_string", &bluechip::GetHandString);

  m.def("calc_all_tables", &rl::bridge::CalcAllTables);
  m.def("generate_deals", &rl::bridge::GenerateDeals);
}

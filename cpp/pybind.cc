//
// Created by qzz on 2023/2/20.
//
#include "bluechip_utils.h"
#include "bridge_actor.h"
#include "bridge_envs.h"
#include "bridge_scoring.h"
#include "bridge_utils.h"
#include "multi_agent_transition_buffer.h"
#include "bridge_state.h"
#include "bridge_thread_loop.h"
#include "rl/context.h"
#include "encode_bridge.h"
#include "rl/model_locker.h"
#include "replay_buffer.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "rl/thread_loop.h"
#include "imp_env.h"
#include "search.h"

namespace py = pybind11;
using namespace bridge_encode;
using namespace rl;
using namespace rl::bridge;
void AccessorTest() {
  torch::Tensor a = torch::tensor({1, 3, 2, 4}, torch::kInt);
  auto accessor = a.accessor<int, 1>();
  for (int i = 0; i < 4; ++i) {
    std::cout << accessor[i] << std::endl;
  }
}
PYBIND11_MODULE(rl_cpp, m) {
  m.def("encode", &Encode, "A function encodes bridge state.");
  m.def("get_imp", &GetImp);

  py::class_<rl::ModelLocker, std::shared_ptr<rl::ModelLocker>>(m,
                                                                "ModelLocker")
      .def(py::init<std::vector<py::object>, const std::string>())
      .def("update_model", &ModelLocker::UpdateModel);

  py::class_<SingleEnvActor, std::shared_ptr<SingleEnvActor>>(
      m, "SingleEnvActor")
      .def(py::init<std::shared_ptr<ModelLocker>>())
      .def("act", &SingleEnvActor::Act);

  py::class_<VecEnvActor, std::shared_ptr<VecEnvActor>>(m, "VecEnvActor")
      .def(py::init<std::shared_ptr<ModelLocker>>())
      .def("act", &VecEnvActor::Act);

  py::class_<ReplayBuffer, std::shared_ptr<ReplayBuffer>>(m, "ReplayBuffer")
      .def(py::init<int, int, int, float, float, float>())
      .def("push", &ReplayBuffer::Push)
      .def("sample", &ReplayBuffer::Sample)
      .def("size", &ReplayBuffer::Size)
      .def("get_all", &ReplayBuffer::GetAll)
      .def("num_add", &ReplayBuffer::NumAdd);
//      .def("dump", &ReplayBuffer::Dump)
//      .def("load", &ReplayBuffer::Load);

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
      .def("trumps", [](const Contract &c) { return static_cast<int>(c.trumps); })
      .def("double_status", [](const Contract &c) { return static_cast<int>(c.double_status); })
      .def_readonly("declarer", &Contract::declarer);

  py::class_<HandEvaluation>(m, "HandEvaluation")
      .def_readonly("high_card_points", &HandEvaluation::high_card_points)
      .def_readonly("length_points", &HandEvaluation::length_points)
      .def_readonly("shortness_points", &HandEvaluation::shortness_points)
      .def_readonly("support_points", &HandEvaluation::support_points)
      .def_readonly("control_count", &HandEvaluation::control_count)
      .def_readonly("length_per_suit", &HandEvaluation::length_per_suit)
      .def("__repr__", &HandEvaluation::ToString);

  m.def("card_string", &CardString);
  m.def("get_card", &Card);
  m.def("bid_string", &BidString);
  m.def("get_bid", &Bid);

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
      .def("observation_tensor", py::overload_cast<>(&BridgeBiddingState::ObservationTensor, py::const_))
      .def("observation_tensor", py::overload_cast<Player>(&BridgeBiddingState::ObservationTensor, py::const_))
      .def("hidden_observation_tensor", &BridgeBiddingState::HiddenObservationTensor)
      .def("observation_tensor_with_hand_evaluation", &BridgeBiddingState::ObservationTensorWithHandEvaluation)
      .def("observation_tensor_with_legal_actions", &BridgeBiddingState::ObservationTensorWithLegalActions)
      .def("cards_tensor", &BridgeBiddingState::CardsTensor)
      .def("final_observation_tensor", py::overload_cast<>(&BridgeBiddingState::FinalObservationTensor, py::const_))
      .def("final_observation_tensor",
           py::overload_cast<Player>(&BridgeBiddingState::FinalObservationTensor, py::const_))
      .def("terminate", &BridgeBiddingState::Terminate)
      .def("legal_actions", &BridgeBiddingState::LegalActions)
      .def("clone", &BridgeBiddingState::Clone)
      .def("get_player_cards", &BridgeBiddingState::GetPlayerCards)
      .def("get_partner_cards", &BridgeBiddingState::GetPartnerCards)
      .def("get_double_dummy_table", &BridgeBiddingState::GetDoubleDummyTable)
      .def("get_hand_evaluation", &BridgeBiddingState::GetHandEvaluation)
      .def("score_for_contracts", &BridgeBiddingState::ScoreForContracts)
      .def("__repr__", &BridgeBiddingState::ToString);

  py::class_<Transition, std::shared_ptr<Transition>>(m, "Transition")
      .def(py::init<>())
      .def_readwrite("obs", &Transition::obs)
      .def_readwrite("reply", &Transition::reply)
      .def_readwrite("reward", &Transition::reward)
      .def_readwrite("terminal", &Transition::terminal)
      .def_readwrite("next_obs", &Transition::next_obs)
      .def("sample_illegal_transitions", &Transition::SampleIllegalTransitions)
      .def("to_dict", &Transition::ToDict);

  py::class_<BridgeTransitionBuffer, std::shared_ptr<BridgeTransitionBuffer>>(m, "BridgeTransitionBuffer")
      .def(py::init<>())
      .def("push_obs_and_reply", &BridgeTransitionBuffer::PushObsAndReply)
      .def("size", &BridgeTransitionBuffer::Size)
      .def("pop_transitions", &BridgeTransitionBuffer::PopTransitions);
//      .def("push_to_replay_buffer", &BridgeTransitionBuffer::PushToReplayBuffer);

  py::class_<MultiAgentTransitionBuffer, std::shared_ptr<MultiAgentTransitionBuffer>>(m, "MultiAgentTransitionBuffer")
      .def(py::init<int>())
      .def("push_obs_and_reply", &MultiAgentTransitionBuffer::PushObsAndReply);

  py::class_<DealManager, std::shared_ptr<DealManager>>(m, "DealManager");

  py::class_<BridgeDealManager, DealManager, std::shared_ptr<BridgeDealManager>>(m, "BridgeDealManager")
      .def(py::init<const std::vector<Cards>,
                    const std::vector<DDT>,
                    const std::vector<int>,
                    int>())
      .def(py::init<const std::vector<Cards>,
                    const std::vector<DDT>,
                    const std::vector<int>>())
      .def("size", &BridgeDealManager::Size)
      .def("next", &BridgeDealManager::Next);

  py::class_<RandomDealManager, DealManager, std::shared_ptr<RandomDealManager>>(m, "RandomDealManager")
      .def(py::init<int>())
      .def("next", &RandomDealManager::Next);

  py::class_<BridgeBiddingEnv, std::shared_ptr<BridgeBiddingEnv>>(m, "BridgeBiddingEnv")
      .def(py::init<
          std::shared_ptr<DealManager>,
          std::vector<int>>())
      .def("reset", &BridgeBiddingEnv::Reset)
      .def("step", &BridgeBiddingEnv::Step)
      .def("returns", &BridgeBiddingEnv::Returns)
      .def("terminated", &BridgeBiddingEnv::Terminated)
      .def("current_player", &BridgeBiddingEnv::CurrentPlayer)
      .def("get_state", &BridgeBiddingEnv::GetState)
      .def("get_num_deals", &BridgeBiddingEnv::GetNumDeals)
      .def("get_feature_size", &BridgeBiddingEnv::GetFeatureSize)
      .def("get_feature", &BridgeBiddingEnv::GetFeature)
      .def("__repr__", &BridgeBiddingEnv::ToString);

  py::class_<BridgeVecEnv, std::shared_ptr<BridgeVecEnv>>(m, "BridgeVecEnv")
      .def(py::init<>())
      .def("push", &BridgeVecEnv::Push, py::keep_alive<1, 2>())
      .def("size", &BridgeVecEnv::Size)
      .def("reset", &BridgeVecEnv::Reset)
      .def("step", &BridgeVecEnv::Step)
      .def("get_envs", &BridgeVecEnv::GetEnvs)
      .def("get_returns", &BridgeVecEnv::GetReturns)
      .def("get_feature", &BridgeVecEnv::GetFeature)
      .def("get_histories", &BridgeVecEnv::GetHistories)
      .def("any_terminated", &BridgeVecEnv::AnyTerminated)
      .def("all_terminated", &BridgeVecEnv::AllTerminated);

  py::class_<BridgeBiddingEnvWrapper, std::shared_ptr<BridgeBiddingEnvWrapper>>(m, "BridgeBiddingEnvWrapper")
      .def(py::init<
          std::shared_ptr<BridgeDealManager>,
          std::vector<int>,
          std::shared_ptr<Replay>>());

  py::class_<BridgeWrapperVecEnv, std::shared_ptr<BridgeWrapperVecEnv>>(m, "BridgeWrapperVecEnv")
      .def(py::init<>())
      .def("push", &BridgeWrapperVecEnv::Push, py::keep_alive<1, 2>());

  py::class_<ImpEnv, std::shared_ptr<ImpEnv>>(m, "ImpEnv")
      .def(py::init<
          std::shared_ptr<BridgeDealManager>,
          std::vector<int>>())
      .def("reset", &ImpEnv::Reset)
      .def("step", &ImpEnv::Step)
      .def("terminated", &ImpEnv::Terminated)
      .def("returns", &ImpEnv::Returns)
      .def("get_acting_player", &ImpEnv::GetActingPlayer)
      .def("get_num_deals", &ImpEnv::GetNumDeals)
      .def("get_feature", &ImpEnv::GetFeature)
      .def("__repr__", &ImpEnv::ToString);

  py::class_<ImpVecEnv, std::shared_ptr<ImpVecEnv>>(m, "ImpVecEnv")
      .def(py::init<>())
      .def("push", &ImpVecEnv::Push, py::keep_alive<1, 2>())
      .def("size", &ImpVecEnv::Size)
      .def("reset", &ImpVecEnv::Reset)
      .def("step", &ImpVecEnv::Step)
      .def("get_feature", &ImpVecEnv::GetFeature)
      .def("get_envs", &ImpVecEnv::GetEnvs)
      .def("all_terminated", &ImpVecEnv::AllTerminated)
      .def("any_terminated", &ImpVecEnv::AnyTerminated);

  py::class_<ImpEnvWrapper, std::shared_ptr<ImpEnvWrapper>>(m, "ImpEnvWrapper")
      .def(py::init<
               std::shared_ptr<BridgeDealManager>,
               const std::vector<int>,
               std::shared_ptr<Replay>>(), py::arg("deal_manager"), py::arg("greedy"),
           py::arg("replay_buffer"))
      .def("get_feature", &ImpEnvWrapper::GetFeature)
      .def("step", &ImpEnvWrapper::Step)
      .def("terminated", &ImpEnvWrapper::Terminated)
      .def("__repr__", &ImpEnvWrapper::ToString)
      .def("reset", &ImpEnvWrapper::Reset);

  py::class_<PVReplay, std::shared_ptr<PVReplay>>(m, "PVReplay")
      .def(py::init<int, int, float, float, int>())
      .def("sample", &PVReplay::Sample)
      .def("num_add", &PVReplay::NumAdd)
      .def("update_priority", &PVReplay::UpdatePriority)
      .def("size", &PVReplay::Size);

  py::class_<Context, std::shared_ptr<Context>>(m, "Context")
      .def(py::init<>())
      .def("push_thread_loop", &Context::PushThreadLoop, py::keep_alive<1, 2>())
      .def("start", &Context::Start)
      .def("pause", &Context::Pause)
      .def("all_paused", &Context::AllPaused)
      .def("resume", &Context::Resume)
      .def("terminate", &Context::Terminate)
      .def("terminated", &Context::Terminated);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

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

  py::class_<BridgeVecEnvThreadLoop, ThreadLoop, std::shared_ptr<BridgeVecEnvThreadLoop>>(m, "BridgeVecEnvThreadLoop")
      .def(py::init<std::shared_ptr<BridgeWrapperVecEnv>, std::shared_ptr<VecEnvActor>>());

  py::class_<DataThreadLoop, ThreadLoop, std::shared_ptr<DataThreadLoop>>(m, "DataThreadLoop")
      .def(py::init<std::shared_ptr<BridgeDealManager>, int>(),
           py::arg("deal_manager"), py::arg("seed"));

  py::class_<VecEnvAllTerminateThreadLoop, ThreadLoop, std::shared_ptr<VecEnvAllTerminateThreadLoop>>
      (m, "VecEnvAllTerminateThreadLoop")
      .def(py::init<std::shared_ptr<VecEnvActor>, std::shared_ptr<BridgeVecEnv>>());

  m.def("make_obs_tensor_dict", &bridge::MakeObsTensorDict);
  m.def("check_prob_not_zero", &rl::utils::CheckProbNotZero);

  m.def("bid_str_to_action", &bluechip::BidStrToAction);
  m.def("bid_action_to_str", &bluechip::BidActionToStr);
  m.def("get_hand_string", &bluechip::GetHandString);

  py::class_<std::mt19937>(m, "RNG")
      .def(py::init<unsigned int>());

  py::class_<Replay, std::shared_ptr<Replay>>(m, "Replay")
      .def(py::init<
               int,
               int,
               float,
               float,
               int>(),
           py::arg("capacity"),
           py::arg("seed"),
           py::arg("alpha"),
           py::arg("beta"),
           py::arg("prefetch"))
      .def("sample", &Replay::Sample)
      .def("num_add", &Replay::NumAdd)
      .def("update_priority", &Replay::UpdatePriority)
      .def("size", &Replay::Size);

  py::class_<ObsBelief, std::shared_ptr<ObsBelief>>(m, "ObsBelief")
      .def(py::init<>())
      .def_readwrite("obs", &ObsBelief::obs)
      .def_readwrite("belief", &ObsBelief::belief);

  py::class_<ObsBeliefReplay, std::shared_ptr<ObsBeliefReplay>>(m, "ObsBeliefReplay")
      .def(py::init<
               int,
               int,
               float,
               float,
               int>(),
           py::arg("capacity"),
           py::arg("seed"),
           py::arg("alpha"),
           py::arg("beta"),
           py::arg("prefetch"))
      .def("sample", &ObsBeliefReplay::Sample)
      .def("num_add", &ObsBeliefReplay::NumAdd)
      .def("update_priority", &ObsBeliefReplay::UpdatePriority)
      .def("size", &ObsBeliefReplay::Size);

  py::class_<BeliefThreadLoop, ThreadLoop, std::shared_ptr<BeliefThreadLoop>>(m, "BeliefThreadLoop")
      .def(py::init<std::shared_ptr<VecEnvActor>,
                    std::shared_ptr<BridgeVecEnv>,
                    std::shared_ptr<ObsBeliefReplay>>());

  py::class_<SearchTransition, std::shared_ptr<SearchTransition>>(m, "SearchTransition")
      .def_readwrite("obs", &SearchTransition::obs)
      .def_readwrite("policy_posterior", &SearchTransition::policy_posterior)
      .def_readwrite("value", &SearchTransition::value);

  py::class_<BeliefModel, std::shared_ptr<BeliefModel>>(m, "BeliefModel")
      .def(py::init<std::shared_ptr<ModelLocker>>());

  py::class_<SearchParams>(m, "SearchParams")
      .def(py::init<>())
      .def_readwrite("num_particles", &SearchParams::num_particles)
      .def_readwrite("temperature", &SearchParams::temperature)
      .def_readwrite("top_k", &SearchParams::top_k)
      .def_readwrite("min_prob", &SearchParams::min_prob)
      .def_readwrite("verbose", &SearchParams::verbose);

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<SearchParams, std::shared_ptr<BeliefModel>, std::shared_ptr<VecEnvActor>>())
      .def("search", &Searcher::Search);

  m.def("accessor_test", &AccessorTest);
}

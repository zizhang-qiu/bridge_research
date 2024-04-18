//
// Created by qzz on 2023/5/21.
//

#ifndef BRIDGE_RESEARCH_CPP_DDS_CALL_H_
#define BRIDGE_RESEARCH_CPP_DDS_CALL_H_
#include "third_party/dds/include/dll.h"
#include "logging.h"
#include "types.h"
#include "bridge_state.h"
#include "bridge_constants.h"
#include <vector>
namespace rl::bridge {
inline constexpr int kMaxDDSBatchSize = 32;

ddTablesRes CalcBatchDDTs(const std::vector<std::vector<Action>> &cards_vector, int mode);

std::vector<Player> GetHolder(const std::vector<Action> &cards);

ddTableDeal Holder2ddTableDeal(const std::vector<Player> &holder);

std::vector<ddTableResults> CalcDDTs(const std::vector<std::vector<Action>> &cards_vector,
                                     int mode,
                                     int num_threads = 1);

ddTableResults CalcOneDeal(const std::vector<Action> &cards);

std::vector<int> ddTableResults2ddt(ddTableResults double_dummy_results);

std::tuple<std::vector<Action>, std::vector<int>> GenerateOneDeal(std::mt19937& rng);
}

#endif //BRIDGE_RESEARCH_CPP_DDS_CALL_H_

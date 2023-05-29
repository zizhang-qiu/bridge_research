//
// Created by qzz on 2023/5/21.
//

#ifndef BRIDGE_RESEARCH_CPP_DDS_H_
#define BRIDGE_RESEARCH_CPP_DDS_H_
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
}

#endif //BRIDGE_RESEARCH_CPP_DDS_H_

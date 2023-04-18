//
// Created by qzz on 2023/4/2.
//
#include <vector>
#include <regex>
#include <string>
#include "utils.h"
#include "bridge_state.h"
#include "third_party/dds/include/dll.h"

#ifndef BRIDGE_RESEARCH_GENERATE_DEALS_H
#define BRIDGE_RESEARCH_GENERATE_DEALS_H
namespace rl::bridge {
using Holder = std::vector<Player>;


inline constexpr int kCalcAllTablesBatchSize = 32;




void DealReturnCode(const int return_code) {
    if (return_code != RETURN_NO_FAULT) {
        char error_message[80];
        ErrorMessage(return_code, error_message);
        std::cerr << utils::StrCat("double_dummy_solver:", error_message) << std::endl;
    }
}

int ExtractParScore(const std::string &par_string) {
    std::regex regex("-?\\d+");
    std::smatch match;
    if (std::regex_search(par_string, match, regex)) {
        return std::stoi(match.str());
    }
    return 0; // or some other default value if no number is found
}

Holder GetHolder(Cards &cards) {
    RL_CHECK_EQ(cards.size(), kNumCards);
    Holder holder(kNumCards);
    for (int i = 0; i < kNumCards; ++i) {
        int this_card = cards[i];
        holder[this_card] = i % kNumPlayers;
    }
    return holder;
}

ddTableDeal HolderToddTableDeal(Holder &holder) {
    ddTableDeal dd_table_deal{};
    for (Suit suit: {Suit::kClubs, Suit::kDiamonds, Suit::kHearts, Suit::kSpades}) {
        for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
            Player player = holder[Card(Suit(suit), rank)];
            dd_table_deal.cards[player][SuitToDDSStrain(suit)] += 1 << (2 + rank);
        }
    }
    return dd_table_deal;
}

DDT ResTableToDDT(int res_table[DDS_STRAINS][DDS_HANDS]) {
    DDT res(kDoubleDummyResultSize);
    for (Denomination denomination: {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
        for (Player player = 0; player < kNumPlayers; ++player) {
            res[denomination * kNumPlayers + player] = res_table[DenominationToDDSStrain(denomination)][player];
        }
    }
    return res;
}

DDT CalcDDTable(Holder &holder) {
    RL_CHECK_EQ(holder.size(), kNumCards);
    auto dd_table_deal = HolderToddTableDeal(holder);
    auto double_dummy_results_ = ddTableResults{};
    SetMaxThreads(0);
    const int return_code = CalcDDtable(
            dd_table_deal, &double_dummy_results_);
    DealReturnCode(return_code);
//    utils::Print2DArray(double_dummy_results_.resTable);
    return ResTableToDDT(double_dummy_results_.resTable);

}

std::tuple<std::vector<DDT>, std::vector<int>> CalcAllTablesOnce(std::vector<Cards> cards_vector) {
    RL_CHECK_LE(cards_vector.size(), kCalcAllTablesBatchSize);
    SetMaxThreads(0);
    int num_batch_deals = cards_vector.size();
    std::vector<Holder> holders(num_batch_deals);
    for (int i = 0; i < num_batch_deals; ++i) {
        holders[i] = GetHolder(cards_vector[i]);
    }
    ddTableDeals dd_table_deals{};
    dd_table_deals.noOfTables = num_batch_deals;
    ddTablesRes dd_tables_res{};

    for (int i = 0; i < num_batch_deals; ++i) {
        auto dd_table_deal = HolderToddTableDeal(holders[i]);
        dd_table_deals.deals[i] = dd_table_deal;
    }

    allParResults pres{};
    int trump_filter[5] = {0, 0, 0, 0, 0};
    const int return_code = CalcAllTables(
            &dd_table_deals, 0, trump_filter, &dd_tables_res, &pres);
    DealReturnCode(return_code);
    std::vector<DDT> res(num_batch_deals);
    std::vector<int> par_scores(num_batch_deals);
    for (int i = 0; i < num_batch_deals; ++i) {
        res[i] = ResTableToDDT(dd_tables_res.results[i].resTable);
//        std::cout << pres.presults[i].parScore[0] << std::endl;
        int par_score = ExtractParScore(pres.presults[i].parScore[0]);
        RL_CHECK_LE(par_score, kMaxScore);
        RL_CHECK_GE(par_score, -kMaxScore);
        par_scores[i] = par_score;
    }


    return std::make_tuple(res, par_scores);

}

std::tuple<std::vector<DDT>, std::vector<int>> CalcAllTables(const std::vector<Cards> &cards_vector) {
    RL_CHECK_TRUE(!cards_vector.empty());
    int num_deals = cards_vector.size();
    int num_batches = static_cast<int>(std::ceil((float)num_deals / kCalcAllTablesBatchSize));
    std::vector<DDT> ddts, batch_ddts;
    std::vector<int> par_scores, batch_par_scores;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    for (int i_batch = 0; i_batch < num_batches; ++i_batch) {
        rl::utils::PrintProgressBar(i_batch, num_batches, start);
        int left = i_batch * kCalcAllTablesBatchSize;
        int right = std::min(left + kCalcAllTablesBatchSize, num_deals);
        std::vector<Cards> batch_cards(cards_vector.begin() + left, cards_vector.begin() + right);
        std::tie(batch_ddts, batch_par_scores) = CalcAllTablesOnce(batch_cards);
        for (int j = 0; j < batch_ddts.size(); ++j) {
            ddts.emplace_back(batch_ddts[j]);
            par_scores.emplace_back(batch_par_scores[j]);
        }
    }
    return std::make_tuple(ddts, par_scores);
}

std::tuple<std::vector<Cards>, std::vector<DDT>, std::vector<int>> GenerateDeals(int num_deals, int seed){
    std::mt19937 rng(seed);
    std::vector<Cards> cards_vector(num_deals);
    for (int i = 0; i < num_deals; ++i) {
        auto cards = rl::utils::Permutation(0, kNumCards, rng);
        cards_vector[i] = cards;
    }

    std::vector<DDT> ddts;
    std::vector<int> par_scores;
    std::tie(ddts, par_scores) = CalcAllTables(cards_vector);
    return std::make_tuple(cards_vector, ddts, par_scores);
}

}
#endif //BRIDGE_RESEARCH_GENERATE_DEALS_H

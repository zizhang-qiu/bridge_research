//
// Created by qzz on 2023/2/25.
//
#include <iostream>
#include "cpp/bridge_lib/bridge_scoring.h"
#include "gtest/gtest.h"
using namespace rl::bridge;
TEST(BridgeScoringTest, ContractMadeNotVulTest) {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpades, kUndoubled};
    const Contract heart_contract{level, kHearts, kUndoubled};
    const int major_contract_points = 30 * level;
    // whether it's small slam or grand slam + part score contract or game bid
    const int major_bonus = std::max(0, level - 5) * 500 + (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamonds, kUndoubled};
    const Contract club_contract{level, kClubs, kUndoubled};
    const int minor_contract_points = 20 * level;
    const int minor_bonus = std::max(0, level - 5) * 500 + (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kUndoubled};
    const int no_trump_contract_points = 30 * level + 10;
    const int no_trump_bonus = std::max(0, level - 5) * 500 + (no_trump_contract_points >= 100 ? 300 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 30 * num_over_tricks;
      const int minor_over_trick_points = 20 * num_over_tricks;
      const int no_trump_over_trick_points = 30 * num_over_tricks;
      EXPECT_EQ(ComputeScore(spade_contract, tricks, false),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(heart_contract, tricks, false),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(diamond_contract, tricks, false),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(club_contract, tricks, false),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(no_trump_contract, tricks, false),
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // doubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpades, kDoubled};
    const Contract heart_contract{level, kHearts, kDoubled};
    const int major_contract_points = 60 * level;
    // doubled contract always get a 50 bonus
    const int major_bonus = 50 + std::max(0, level - 5) * 500 + (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamonds, kDoubled};
    const Contract club_contract{level, kClubs, kDoubled};
    const int minor_contract_points = 40 * level;
    const int minor_bonus = 50 + std::max(0, level - 5) * 500 + (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kDoubled};
    const int no_trump_contract_points = 60 * level + 20;
    const int no_trump_bonus = 50 + std::max(0, level - 5) * 500 + (no_trump_contract_points >= 100 ? 300 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 100 * num_over_tricks;
      const int minor_over_trick_points = 100 * num_over_tricks;
      const int no_trump_over_trick_points = 100 * num_over_tricks;
      EXPECT_EQ(ComputeScore(spade_contract, tricks, false),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(heart_contract, tricks, false),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(diamond_contract, tricks, false),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(club_contract, tricks, false),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(no_trump_contract, tricks, false),
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpades, kRedoubled};
    const Contract heart_contract{level, kHearts, kRedoubled};
    const int major_contract_points = 120 * level;
    // redoubled contract always get a 100 bonus
    const int major_bonus = 100 + std::max(0, level - 5) * 500 + (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamonds, kRedoubled};
    const Contract club_contract{level, kClubs, kRedoubled};
    const int minor_contract_points = 80 * level;
    const int minor_bonus = 100 + std::max(0, level - 5) * 500 + (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kRedoubled};
    const int no_trump_contract_points = 120 * level + 40;
    const int no_trump_bonus = 100 + std::max(0, level - 5) * 500 + (no_trump_contract_points >= 100 ? 300 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 200 * num_over_tricks;
      const int minor_over_trick_points = 200 * num_over_tricks;
      const int no_trump_over_trick_points = 200 * num_over_tricks;
      EXPECT_EQ(ComputeScore(spade_contract, tricks, false),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(heart_contract, tricks, false),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(diamond_contract, tricks, false),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(club_contract, tricks, false),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(no_trump_contract, tricks, false),
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }
}

TEST(BridgeScoringTest, ContractMadeVulTest) {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpades, kUndoubled};
    const Contract heart_contract{level, kHearts, kUndoubled};
    const int major_contract_points = 30 * level;
    // whether it's small slam or grand slam + part score contract or game bid
    const int major_bonus = std::max(0, level - 5) * 750 + (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamonds, kUndoubled};
    const Contract club_contract{level, kClubs, kUndoubled};
    const int minor_contract_points = 20 * level;
    const int minor_bonus = std::max(0, level - 5) * 750 + (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kUndoubled};
    const int no_trump_contract_points = 30 * level + 10;
    const int no_trump_bonus = std::max(0, level - 5) * 750 + (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 30 * num_over_tricks;
      const int minor_over_trick_points = 20 * num_over_tricks;
      const int no_trump_over_trick_points = 30 * num_over_tricks;
      EXPECT_EQ(ComputeScore(spade_contract, tricks, true),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(heart_contract, tricks, true),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(diamond_contract, tricks, true),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(club_contract, tricks, true),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(no_trump_contract, tricks, true),
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // doubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpades, kDoubled};
    const Contract heart_contract{level, kHearts, kDoubled};
    const int major_contract_points = 60 * level;
    // doubled contract always get a 50 bonus
    const int major_bonus = 50 + std::max(0, level - 5) * 750 + (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamonds, kDoubled};
    const Contract club_contract{level, kClubs, kDoubled};
    const int minor_contract_points = 40 * level;
    const int minor_bonus = 50 + std::max(0, level - 5) * 750 + (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kDoubled};
    const int no_trump_contract_points = 60 * level + 20;
    const int no_trump_bonus = 50 + std::max(0, level - 5) * 750 + (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 200 * num_over_tricks;
      const int minor_over_trick_points = 200 * num_over_tricks;
      const int no_trump_over_trick_points = 200 * num_over_tricks;
      EXPECT_EQ(ComputeScore(spade_contract, tricks, true),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(heart_contract, tricks, true),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(diamond_contract, tricks, true),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(club_contract, tricks, true),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(no_trump_contract, tricks, true),
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpades, kRedoubled};
    const Contract heart_contract{level, kHearts, kRedoubled};
    const int major_contract_points = 120 * level;
    // redoubled contract always get a 100 bonus
    const int major_bonus = 100 + std::max(0, level - 5) * 750 + (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamonds, kRedoubled};
    const Contract club_contract{level, kClubs, kRedoubled};
    const int minor_contract_points = 80 * level;
    const int minor_bonus = 100 + std::max(0, level - 5) * 750 + (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kRedoubled};
    const int no_trump_contract_points = 120 * level + 40;
    const int no_trump_bonus = 100 + std::max(0, level - 5) * 750 + (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 400 * num_over_tricks;
      const int minor_over_trick_points = 400 * num_over_tricks;
      const int no_trump_over_trick_points = 400 * num_over_tricks;
      EXPECT_EQ(ComputeScore(spade_contract, tricks, true),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(heart_contract, tricks, true),
                major_contract_points + major_over_trick_points + major_bonus);
      EXPECT_EQ(ComputeScore(diamond_contract, tricks, true),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(club_contract, tricks, true),
                minor_contract_points + minor_over_trick_points + minor_bonus);
      EXPECT_EQ(ComputeScore(no_trump_contract, tricks, true),
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }
}

TEST(BridgeScoringTest, ContractDefeatedNotVulTest) {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      const Contract contract{level, trump, kUndoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = 50 * num_under_tricks;
        EXPECT_EQ(ComputeScore(contract, tricks, false), -penalty);
      }
    }
  }

  // doubled
  constexpr int kPenalty[] = {0, 100, 300, 500, 800};
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      const Contract contract{level, trump, kDoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4) ? kPenalty[num_under_tricks] : 300 * (num_under_tricks - 4) + 800;
        EXPECT_EQ(ComputeScore(contract, tricks, false), -penalty);
      }
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      const Contract contract{level, trump, kRedoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty =
            (num_under_tricks <= 4) ? (kPenalty[num_under_tricks] * 2) : 600 * (num_under_tricks - 4) + 1600;
        EXPECT_EQ(ComputeScore(contract, tricks, false), -penalty);
      }
    }
  }

}

TEST(BridgeScoringTest, ContractDefeatedVulTest) {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      const Contract contract{level, trump, kUndoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = 100 * num_under_tricks;
        EXPECT_EQ(ComputeScore(contract, tricks, true), -penalty);
      }
    }
  }

  // doubled
  constexpr int kPenalty[] = {0, 200, 500, 800, 1100};
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      const Contract contract{level, trump, kDoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4) ? kPenalty[num_under_tricks] : 300 * (num_under_tricks - 4) + 1100;
        EXPECT_EQ(ComputeScore(contract, tricks, true), -penalty);
      }
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      const Contract contract{level, trump, kRedoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty =
            (num_under_tricks <= 4) ? (kPenalty[num_under_tricks] * 2) : 600 * (num_under_tricks - 4) + 2200;
        EXPECT_EQ(ComputeScore(contract, tricks, true), -penalty);
      }
    }
  }
}

int VanillaImp(int score1, int score2) {
  int difference = score1 - score2;
  int sign = difference == 0 ? 0 : (difference > 0 ? 1 : -1);
  int abs_score = std::abs(difference);
  if (0 <= abs_score && abs_score <= 10) {
    return sign * 0;
  }
  if (20 <= abs_score && abs_score <= 40) {
    return sign * 1;
  }
  if (50 <= abs_score && abs_score <= 80) {
    return sign * 2;
  }
  if (90 <= abs_score && abs_score <= 120) {
    return sign * 3;
  }
  if (130 <= abs_score && abs_score <= 160) {
    return sign * 4;
  }
  if (170 <= abs_score && abs_score <= 210) {
    return sign * 5;
  }
  if (220 <= abs_score && abs_score <= 260) {
    return sign * 6;
  }
  if (270 <= abs_score && abs_score <= 310) {
    return sign * 7;
  }
  if (320 <= abs_score && abs_score <= 360) {
    return sign * 8;
  }
  if (370 <= abs_score && abs_score <= 420) {
    return sign * 9;
  }
  if (430 <= abs_score && abs_score <= 490) {
    return sign * 10;
  }
  if (500 <= abs_score && abs_score <= 590) {
    return sign * 11;
  }
  if (600 <= abs_score && abs_score <= 740) {
    return sign * 12;
  }
  if (750 <= abs_score && abs_score <= 890) {
    return sign * 13;
  }
  if (900 <= abs_score && abs_score <= 1090) {
    return sign * 14;
  }
  if (1100 <= abs_score && abs_score <= 1290) {
    return sign * 15;
  }
  if (1300 <= abs_score && abs_score <= 1490) {
    return sign * 16;
  }
  if (1500 <= abs_score && abs_score <= 1740) {
    return sign * 17;
  }
  if (1750 <= abs_score && abs_score <= 1990) {
    return sign * 18;
  }
  if (2000 <= abs_score && abs_score <= 2240) {
    return sign * 19;
  }
  if (2250 <= abs_score && abs_score <= 2490) {
    return sign * 20;
  }
  if (2500 <= abs_score && abs_score <= 2990) {
    return sign * 21;
  }
  if (3000 <= abs_score && abs_score <= 3490) {
    return sign * 22;
  }
  if (3500 <= abs_score && abs_score <= 3990) {
    return sign * 23;
  } else {
    return sign * 24;
  }
}

TEST(BridgeScoringTest, IMPTest) {
  auto all_scores = AllScores();
  int size = all_scores.size();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      EXPECT_EQ(VanillaImp(all_scores[i], all_scores[j]), GetImp(all_scores[i], all_scores[j]));
    }
  }
}
//
// Created by qzz on 2023/3/4.
//


#ifndef BRIDGE_RESEARCH_RL_BASE_H
#define BRIDGE_RESEARCH_RL_BASE_H
#include "torch/torch.h"
#include "tensor_dict.h"
#include "types.h"
namespace rl {
class State{
  public:
  virtual ~State() = default;
  // Returns current player. Player numbers start from 0.
  // Negative numbers are for chance (-1) or simultaneous (-2).
  // kTerminalPlayerId should be returned on a TerminalNode().
  virtual Player CurrentPlayer() const = 0;
  virtual void ApplyAction(Action action_id) = 0;
  virtual std::vector<Action> LegalActions() const = 0;

  // Returns a string representation of the state. Also used as in the default
  // implementation of operator==.
  virtual std::string ToString() const = 0;

  std::vector<Action> History() const {
    std::vector<Action> history;
    history.reserve(history_.size());
    for (auto& h : history_) history.push_back(h.action);
    return history;
  };
 protected:
  // Information that changes over the course of the game.
  std::vector<PlayerAction> history_;
};

class Env {
 public:
  Env() = default;

  ~Env() = default;

  virtual bool Reset() = 0;

  virtual void Step(const TensorDict &reply) = 0;

  [[nodiscard]] virtual std::vector<float> Returns() const = 0;

  [[nodiscard]] virtual TensorDict GetFeature() const = 0;

  [[nodiscard]] virtual bool Terminated() const = 0;

  [[nodiscard]] virtual Player CurrentPlayer() const = 0;
};

class Actor {
 public:
  Actor() = default;

  ~Actor() = default;

  virtual TensorDict Act(const TensorDict &obs) = 0;
};
}
#endif //BRIDGE_RESEARCH_RL_BASE_H

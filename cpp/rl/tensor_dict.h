//
// Created by qzz on 2023/4/21.
//

#ifndef BRIDGE_RESEARCH_CPP_TENSOR_DICT_H_
#define BRIDGE_RESEARCH_CPP_TENSOR_DICT_H_
#include <torch/extension.h>
#include <unordered_map>
namespace rl{
using TensorDict = std::unordered_map<std::string, torch::Tensor>;
using TorchTensorDict = torch::Dict<std::string, torch::Tensor>;

namespace tensor_dict{
inline void CompareShape(const TensorDict& src, const TensorDict& dest) {
  if (src.size() != dest.size()) {
    std::cout << "src.size()[" << src.size() << "] != dest.size()[" << dest.size() << "]"
              << std::endl;
    std::cout << "src keys: ";
    for (const auto& p : src)
      std::cout << p.first << " ";
    std::cout << "dest keys: ";
    for (const auto& p : dest)
      std::cout << p.first << " ";
    std::cout << std::endl;
    assert(false);
  }

  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    const auto& destTensor = dest.at(name);
    if (destTensor.sizes() != srcTensor.sizes()) {
      std::cout << name << ", dstSize: " << destTensor.sizes()
                << ", srcSize: " << srcTensor.sizes() << std::endl;
      assert(false);
    }
  }
}

inline void Copy(const TensorDict& src, TensorDict& dest) {
  CompareShape(src, dest);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    destTensor.copy_(srcTensor);
  }
}

inline void Copy(const TensorDict& src, TensorDict& dest, const torch::Tensor& index) {
  assert(src.size() == dest.size());
  assert(index.size(0) > 0);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    assert(destTensor.dtype() == srcTensor.dtype());
    assert(index.size(0) == srcTensor.size(0));
    destTensor.index_copy_(0, index, srcTensor);
  }
}

inline bool Equal(const TensorDict& d0, const TensorDict& d1) {
  if (d0.size() != d1.size()) {
    return false;
  }

  for (const auto& name2tensor : d0) {
    auto key = name2tensor.first;
    if ((d1.at(key) != name2tensor.second).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

/*
 * indexes into a TensorDict
 */
inline TensorDict Index(const TensorDict& batch, size_t i) {
  TensorDict result;
  for (const auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second[i]});
  }
  return result;
}

inline TensorDict Narrow(
    const TensorDict& batch, size_t dim, size_t i, size_t len, bool squeeze) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    auto t = name2tensor.second.narrow(dim, i, len);
    if (squeeze) {
      assert(len == 1);
      t = t.squeeze(dim);
    }
    result.insert({name2tensor.first, std::move(t)});
  }
  return result;
}

inline TensorDict Clone(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, name2tensor.second.clone()});
  }
  return output;
}

inline TensorDict ZerosLike(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, torch::zeros_like(name2tensor.second)});
  }
  return output;
}

// TODO: rewrite the above functions with this template
template <typename Func>
inline TensorDict Apply(TensorDict& dict, Func f) {
  TensorDict output;
  for (const auto& name2tensor : dict) {
    auto tensor = f(name2tensor.second);
    output.insert({name2tensor.first, tensor});
  }
  return output;
}

inline TensorDict Stack(const std::vector<TensorDict>& vec, int stackdim) {
  assert(vec.size() >= 1);
  TensorDict ret;
  for (auto& name2tensor : vec[0]) {
    std::vector<torch::Tensor> buffer(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      buffer[i] = vec[i].at(name2tensor.first);
    }
    ret[name2tensor.first] = torch::stack(buffer, stackdim);
  }
  return ret;
}

inline TensorDict FromIValue(
    const torch::jit::IValue& value, torch::DeviceType device, bool detach) {
  std::unordered_map<std::string, torch::Tensor> map;
  auto dict = value.toGenericDict();
  // auto ivalMap = dict->elements();
  for (auto& name2tensor : dict) {
    auto name = name2tensor.key().toString();
    torch::Tensor tensor = name2tensor.value().toTensor();
    tensor = tensor.to(device);
    if (detach) {
      tensor = tensor.detach();
    }
    map.insert({name->string(), tensor});
  }
  return map;
}

// TODO: this may be simplified with constructor in the future version
inline torch::jit::IValue ToIValue(
    const TensorDict& tensorDict, const torch::Device& device) {
  torch::Dict<std::string, torch::Tensor> dict;
  for (const auto& name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return torch::jit::IValue(dict);
}

inline TorchTensorDict ToTorchDict(const TensorDict& tensorDict,
                                             const torch::Device& device) {
  TorchTensorDict dict;
  for (const auto& name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return dict;
}
}
}
#endif //BRIDGE_RESEARCH_CPP_TENSOR_DICT_H_

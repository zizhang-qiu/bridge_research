//
// Created by qzz on 2023/4/17.
//
#include <cstdio>
#ifndef BRIDGE_RESEARCH_SPAN_H
#define BRIDGE_RESEARCH_SPAN_H
namespace rl::utils {
template <typename T>
class Span {
public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator = T*;
  using const_iterator = const T*;
  using size_type = std::size_t;

  constexpr Span() noexcept : data_(nullptr), size_(0) {}

  constexpr Span(pointer data, size_type size) noexcept : data_(data), size_(size) {}

  template <typename U, std::size_t N>
  constexpr Span(U (&arr)[N]) noexcept : data_(arr), size_(N) {}

  template <typename Container, typename = std::enable_if_t<
                                    !std::is_pointer_v<Container> &&
                                    !std::is_same_v<Container, Span>>>
  constexpr Span(Container& cont) noexcept
      : data_(cont.data()), size_(cont.size()) {}

  template <typename Container, typename = std::enable_if_t<
                                    !std::is_pointer_v<Container> &&
                                    !std::is_same_v<Container, Span>>>
  constexpr Span(const Container& cont) noexcept
      : data_(cont.data()), size_(cont.size()) {}

  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + size_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type size_bytes() const noexcept { return size_ * sizeof(T); }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr pointer data() const noexcept { return data_; }
  constexpr reference operator[](size_type index) const noexcept { return data_[index]; }

private:
  pointer data_;
  size_type size_;
};
} // namespace rl::utils
#endif // BRIDGE_RESEARCH_SPAN_H

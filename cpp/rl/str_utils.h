//
// Created by qzz on 2023/4/17.
//

#ifndef BRIDGE_RESEARCH_RL_STR_UTILS_H
#define BRIDGE_RESEARCH_RL_STR_UTILS_H
#include <memory>
#include <stdexcept>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#undef snprintf
namespace rl::utils {

namespace str_utils::internal {
// Declaration for an array of bitfields holding character information.
extern const unsigned char kPropertyBits[256];

// Declaration for the array of characters to upper-case characters.
extern const char kToUpper[256];

// Declaration for the array of characters to lower-case characters.
extern const char kToLower[256];
}

template<typename... Args>
std::string StrFormat(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return {buf.get(), buf.get() + size - 1}; // We don't want the '\0' inside
}

template<typename T>
void StrAppendImpl(std::string *str, T arg) {
  std::ostringstream ss;
  ss << arg;
  str->append(ss.str());
}

template<typename T, typename... Args>
void StrAppendImpl(std::string *str, T arg, Args... args) {
  StrAppendImpl(str, arg);
  StrAppendImpl(str, args...);
}

template<typename... Args>
void StrAppend(std::string *str, Args... args) {
  StrAppendImpl(str, args...);
}

inline void StrAppend(std::string *) {}

inline std::string StrJoin(const std::vector<std::string> &values, const std::string &delimiter) {
  std::ostringstream os;
  auto it = std::begin(values);
  const auto end = std::end(values);
  if (it != end) {
    os << *it++;
  }
  while (it != end) {
    os << delimiter << *it++;
  }
  return os.str();
}

inline std::vector<std::string> StrSplit(const std::string &str, char delimiter) {
  std::vector<std::string> result;
  std::istringstream iss(str);
  std::string token;

  while (std::getline(iss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}

inline std::string &LTrim(std::string &str, const std::string &chars = "\t\n\v\f\r ") {
  str.erase(0, str.find_first_not_of(chars));
  return str;
}

inline std::string &RTrim(std::string &str, const std::string &chars = "\t\n\v\f\r ") {
  str.erase(str.find_last_not_of(chars) + 1);
  return str;
}

inline std::string &Trim(std::string &str, const std::string &chars = "\t\n\v\f\r ") {
  return LTrim(RTrim(str, chars), chars);
}

inline bool StrContains(const std::string &str, const std::string &substr) {
  return str.find(substr) != std::string::npos;
}

inline bool StrContains(const std::string &str, char substr) {
  return str.find(substr) != std::string::npos;
}

template<class... Args>
inline std::string StrCat(const Args &...args) {
  using Expander = int[];
  std::stringstream ss;
  (void) Expander{0, (void(ss << args), 0)...};
  return ss.str();
}

inline bool StartsWith(const std::string &text,
                       const std::string &prefix) {
  return prefix.empty() ||
      (text.size() >= prefix.size() &&
          memcmp(text.data(), prefix.data(), prefix.size()) == 0);
}

// ascii_isalpha()
//
// Determines whether the given character is an alphabetic character.
inline bool ascii_isalpha(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x01) != 0;
}

// ascii_isalnum()
//
// Determines whether the given character is an alphanumeric character.
inline bool ascii_isalnum(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x04) != 0;
}

// ascii_isspace()
//
// Determines whether the given character is a whitespace character (space,
// tab, vertical tab, formfeed, linefeed, or carriage return).
inline bool ascii_isspace(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x08) != 0;
}

// ascii_ispunct()
//
// Determines whether the given character is a punctuation character.
inline bool ascii_ispunct(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x10) != 0;
}

// ascii_isblank()
//
// Determines whether the given character is a blank character (tab or space).
inline bool ascii_isblank(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x20) != 0;
}

// ascii_iscntrl()
//
// Determines whether the given character is a control character.
inline bool ascii_iscntrl(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x40) != 0;
}

// ascii_isxdigit()
//
// Determines whether the given character can be represented as a hexadecimal
// digit character (i.e. {0-9} or {A-F}).
inline bool ascii_isxdigit(unsigned char c) {
  return (str_utils::internal::kPropertyBits[c] & 0x80) != 0;
}

// ascii_isdigit()
//
// Determines whether the given character can be represented as a decimal
// digit character (i.e. {0-9}).
inline bool ascii_isdigit(unsigned char c) { return c >= '0' && c <= '9'; }

// ascii_isprint()
//
// Determines whether the given character is printable, including spaces.
inline bool ascii_isprint(unsigned char c) { return c >= 32 && c < 127; }

// ascii_isgraph()
//
// Determines whether the given character has a graphical representation.
inline bool ascii_isgraph(unsigned char c) { return c > 32 && c < 127; }

// ascii_isupper()
//
// Determines whether the given character is uppercase.
inline bool ascii_isupper(unsigned char c) { return c >= 'A' && c <= 'Z'; }

// ascii_islower()
//
// Determines whether the given character is lowercase.
inline bool ascii_islower(unsigned char c) { return c >= 'a' && c <= 'z'; }

// ascii_isascii()
//
// Determines whether the given character is ASCII.
inline bool ascii_isascii(unsigned char c) { return c < 128; }

// ascii_tolower()
//
// Returns an ASCII character, converting to lowercase if uppercase is
// passed. Note that character values > 127 are simply returned.
inline char ascii_tolower(unsigned char c) {
  return str_utils::internal::kToLower[c];
}

// Converts the characters in `s` to lowercase, changing the contents of `s`.
void AsciiStrToLower(std::string *s);

// Creates a lowercase string from a given absl::string_view.
inline std::string AsciiStrToLower(std::string_view s) {
  std::string result(s);
  AsciiStrToLower(&result);
  return result;
}

// ascii_toupper()
//
// Returns the ASCII character, converting to upper-case if lower-case is
// passed. Note that characters values > 127 are simply returned.
inline char ascii_toupper(unsigned char c) {
  return str_utils::internal::kToUpper[c];
}

// Converts the characters in `s` to uppercase, changing the contents of `s`.
void AsciiStrToUpper(std::string *s);

// Creates an uppercase string from a given absl::string_view.
inline std::string AsciiStrToUpper(std::string_view s) {
  std::string result(s);
  AsciiStrToUpper(&result);
  return result;
}

} // namespace rl::utils

#endif // BRIDGE_RESEARCH_RL_STR_UTILS_H

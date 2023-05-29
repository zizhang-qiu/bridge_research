//
// Created by qzz on 2023/5/19.
//
#include <algorithm>
#include "gtest/gtest.h"
#include "../rl/str_utils.h"
using namespace rl::utils;
TEST(AsciiIsFoo, All) {
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
      EXPECT_TRUE(::ascii_isalpha(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isalpha(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if ((c >= '0' && c <= '9'))
      EXPECT_TRUE(::ascii_isdigit(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isdigit(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (::ascii_isalpha(c) || ::ascii_isdigit(c))
      EXPECT_TRUE(::ascii_isalnum(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isalnum(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i != '\0' && strchr(" \r\n\t\v\f", i))
      EXPECT_TRUE(::ascii_isspace(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isspace(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i >= 32 && i < 127)
      EXPECT_TRUE(::ascii_isprint(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isprint(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (::ascii_isprint(c) && !::ascii_isspace(c) &&
        !::ascii_isalnum(c)) {
      EXPECT_TRUE(::ascii_ispunct(c)) << ": failed on " << c;
    } else {
      EXPECT_TRUE(!::ascii_ispunct(c)) << ": failed on " << c;
    }
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i == ' ' || i == '\t')
      EXPECT_TRUE(::ascii_isblank(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isblank(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i < 32 || i == 127)
      EXPECT_TRUE(::ascii_iscntrl(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_iscntrl(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (::ascii_isdigit(c) || (i >= 'A' && i <= 'F') ||
        (i >= 'a' && i <= 'f')) {
      EXPECT_TRUE(::ascii_isxdigit(c)) << ": failed on " << c;
    } else {
      EXPECT_TRUE(!::ascii_isxdigit(c)) << ": failed on " << c;
    }
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i > 32 && i < 127)
      EXPECT_TRUE(::ascii_isgraph(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isgraph(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i >= 'A' && i <= 'Z')
      EXPECT_TRUE(::ascii_isupper(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_isupper(c)) << ": failed on " << c;
  }
  for (int i = 0; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    if (i >= 'a' && i <= 'z')
      EXPECT_TRUE(::ascii_islower(c)) << ": failed on " << c;
    else
      EXPECT_TRUE(!::ascii_islower(c)) << ": failed on " << c;
  }
  for (unsigned char c = 0; c < 128; c++) {
    EXPECT_TRUE(::ascii_isascii(c)) << ": failed on " << c;
  }
  for (int i = 128; i < 256; i++) {
    const auto c = static_cast<unsigned char>(i);
    EXPECT_TRUE(!::ascii_isascii(c)) << ": failed on " << c;
  }
}

TEST(AsciiStrTo, Lower) {
  const char buf[] = "ABCDEF";
  const std::string str("GHIJKL");
  const std::string str2("MNOPQR");
  const std::string_view sp(str2);
  std::string mutable_str("_`?@[{AMNOPQRSTUVWXYZ");

  EXPECT_EQ("abcdef", ::AsciiStrToLower(buf));
  EXPECT_EQ("ghijkl", ::AsciiStrToLower(str));
  EXPECT_EQ("mnopqr", ::AsciiStrToLower(sp));

  ::AsciiStrToLower(&mutable_str);
  EXPECT_EQ("_`?@[{amnopqrstuvwxyz", mutable_str);

  char mutable_buf[] = "Mutable";
  std::transform(mutable_buf, mutable_buf + strlen(mutable_buf),
                 mutable_buf, ::ascii_tolower);
  EXPECT_STREQ("mutable", mutable_buf);
}

TEST(AsciiStrTo, Upper) {
  const char buf[] = "abcdef";
  const std::string str("ghijkl");
  const std::string str2("_`?@[{amnopqrstuvwxyz");
  const std::string_view sp(str2);

  EXPECT_EQ("ABCDEF", ::AsciiStrToUpper(buf));
  EXPECT_EQ("GHIJKL", ::AsciiStrToUpper(str));
  EXPECT_EQ("_`?@[{AMNOPQRSTUVWXYZ", ::AsciiStrToUpper(sp));

  char mutable_buf[] = "Mutable";
  std::transform(mutable_buf, mutable_buf + strlen(mutable_buf),
                 mutable_buf, ::ascii_toupper);
  EXPECT_STREQ("MUTABLE", mutable_buf);
}

TEST(StrCat, Ints) {
  const short s = -1;  // NOLINT(runtime/int)
  const uint16_t us = 2;
  const int i = -3;
  const unsigned int ui = 4;
  const long l = -5;                 // NOLINT(runtime/int)
  const unsigned long ul = 6;        // NOLINT(runtime/int)
  const long long ll = -7;           // NOLINT(runtime/int)
  const unsigned long long ull = 8;  // NOLINT(runtime/int)
  const ptrdiff_t ptrdiff = -9;
  const size_t size = 10;
  const intptr_t intptr = -12;
  const uintptr_t uintptr = 13;
  std::string answer;
  answer = ::StrCat(s, us);
  EXPECT_EQ(answer, "-12");
  answer = ::StrCat(i, ui);
  EXPECT_EQ(answer, "-34");
  answer = ::StrCat(l, ul);
  EXPECT_EQ(answer, "-56");
  answer = ::StrCat(ll, ull);
  EXPECT_EQ(answer, "-78");
  answer = ::StrCat(ptrdiff, size);
  EXPECT_EQ(answer, "-910");
  answer = ::StrCat(ptrdiff, intptr);
  EXPECT_EQ(answer, "-9-12");
  answer = ::StrCat(uintptr, 0);
  EXPECT_EQ(answer, "130");
}

TEST(StrCat, Basics) {
  std::string result;

  std::string strs[] = {"Hello", "Cruel", "World"};

  std::string stdstrs[] = {
      "std::Hello",
      "std::Cruel",
      "std::World"
  };

  std::string_view pieces[] = {"Hello", "Cruel", "World"};

  const char* c_strs[] = {
      "Hello",
      "Cruel",
      "World"
  };

  int32_t i32s[] = {'H', 'C', 'W'};
  uint64_t ui64s[] = {12345678910LL, 10987654321LL};

  EXPECT_EQ(::StrCat(), "");

  result = ::StrCat(false, true, 2, 3);
  EXPECT_EQ(result, "0123");

  result = ::StrCat(-1);
  EXPECT_EQ(result, "-1");

  result = ::StrCat(strs[1], pieces[2]);
  EXPECT_EQ(result, "CruelWorld");

  result = ::StrCat(stdstrs[1], " ", stdstrs[2]);
  EXPECT_EQ(result, "std::Cruel std::World");

  result = ::StrCat(strs[0], ", ", pieces[2]);
  EXPECT_EQ(result, "Hello, World");

  result = ::StrCat(strs[0], ", ", strs[1], " ", strs[2], "!");
  EXPECT_EQ(result, "Hello, Cruel World!");

  result = ::StrCat(pieces[0], ", ", pieces[1], " ", pieces[2]);
  EXPECT_EQ(result, "Hello, Cruel World");

  result = ::StrCat(c_strs[0], ", ", c_strs[1], " ", c_strs[2]);
  EXPECT_EQ(result, "Hello, Cruel World");

  result = ::StrCat("ASCII ", i32s[0], ", ", i32s[1], " ", i32s[2], "!");
  EXPECT_EQ(result, "ASCII 72, 67 87!");

  result = ::StrCat(ui64s[0], ", ", ui64s[1], "!");
  EXPECT_EQ(result, "12345678910, 10987654321!");

  std::string one =
      "1";  // Actually, it's the size of this string that we want; a
  // 64-bit build distinguishes between size_t and uint64_t,
  // even though they're both unsigned 64-bit values.
  result = ::StrCat("And a ", one.size(), " and a ",
                        &result[2] - &result[0], " and a ", one, " 2 3 4", "!");
  EXPECT_EQ(result, "And a 1 and a 2 and a 1 2 3 4!");

  // result = ::StrCat("Single chars won't compile", '!');
  // result = ::StrCat("Neither will nullptrs", nullptr);
  result =
      ::StrCat("To output a char by ASCII/numeric value, use +: ", '!' + 0);
  EXPECT_EQ(result, "To output a char by ASCII/numeric value, use +: 33");


  result = ::StrCat(1, 2, 333, 4444, 55555, 666666, 7777777, 88888888,
                        999999999);
  EXPECT_EQ(result, "12333444455555666666777777788888888999999999");
}

TEST(StrCat, CornerCases) {
  std::string result;

  result = ::StrCat("");  // NOLINT
  EXPECT_EQ(result, "");
  result = ::StrCat("", "");
  EXPECT_EQ(result, "");
  result = ::StrCat("", "", "");
  EXPECT_EQ(result, "");
  result = ::StrCat("", "", "", "");
  EXPECT_EQ(result, "");
  result = ::StrCat("", "", "", "", "");
  EXPECT_EQ(result, "");
}

TEST(StrAppend, Basics) {
  std::string result = "existing text";

  std::string strs[] = {"Hello", "Cruel", "World"};

  std::string stdstrs[] = {
    "std::Hello",
    "std::Cruel",
    "std::World"
  };

  std::string_view pieces[] = {"Hello", "Cruel", "World"};

  const char* c_strs[] = {
    "Hello",
    "Cruel",
    "World"
  };

  int32_t i32s[] = {'H', 'C', 'W'};
  uint64_t ui64s[] = {12345678910LL, 10987654321LL};

  std::string::size_type old_size = result.size();
  ::StrAppend(&result);
  EXPECT_EQ(result.size(), old_size);

  old_size = result.size();
  ::StrAppend(&result, strs[0]);
  EXPECT_EQ(result.substr(old_size), "Hello");

  old_size = result.size();
  ::StrAppend(&result, strs[1], pieces[2]);
  EXPECT_EQ(result.substr(old_size), "CruelWorld");

  old_size = result.size();
  ::StrAppend(&result, stdstrs[0], ", ", pieces[2]);
  EXPECT_EQ(result.substr(old_size), "std::Hello, World");

  old_size = result.size();
  ::StrAppend(&result, strs[0], ", ", stdstrs[1], " ", strs[2], "!");
  EXPECT_EQ(result.substr(old_size), "Hello, std::Cruel World!");

  old_size = result.size();
  ::StrAppend(&result, pieces[0], ", ", pieces[1], " ", pieces[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World");

  old_size = result.size();
  ::StrAppend(&result, c_strs[0], ", ", c_strs[1], " ", c_strs[2]);
  EXPECT_EQ(result.substr(old_size), "Hello, Cruel World");

  old_size = result.size();
  ::StrAppend(&result, "ASCII ", i32s[0], ", ", i32s[1], " ", i32s[2], "!");
  EXPECT_EQ(result.substr(old_size), "ASCII 72, 67 87!");

  old_size = result.size();
  ::StrAppend(&result, ui64s[0], ", ", ui64s[1], "!");
  EXPECT_EQ(result.substr(old_size), "12345678910, 10987654321!");

  std::string one =
      "1";  // Actually, it's the size of this string that we want; a
            // 64-bit build distinguishes between size_t and uint64_t,
            // even though they're both unsigned 64-bit values.
  old_size = result.size();
  ::StrAppend(&result, "And a ", one.size(), " and a ",
                  &result[2] - &result[0], " and a ", one, " 2 3 4", "!");
  EXPECT_EQ(result.substr(old_size), "And a 1 and a 2 and a 1 2 3 4!");

  // result = ::StrCat("Single chars won't compile", '!');
  // result = ::StrCat("Neither will nullptrs", nullptr);
  old_size = result.size();
  ::StrAppend(&result,
                  "To output a char by ASCII/numeric value, use +: ", '!' + 0);
  EXPECT_EQ(result.substr(old_size),
            "To output a char by ASCII/numeric value, use +: 33");

  // Test 9 arguments, the old maximum
  old_size = result.size();
  ::StrAppend(&result, 1, 22, 333, 4444, 55555, 666666, 7777777, 88888888,
                  9);
  EXPECT_EQ(result.substr(old_size), "1223334444555556666667777777888888889");

  // No limit thanks to C++11's variadic templates
  old_size = result.size();
  ::StrAppend(
      &result, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                           //
      "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",  //
      "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",  //
      "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",  //
      "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",  //
      "No limit thanks to C++11's variadic templates");
  EXPECT_EQ(result.substr(old_size),
            "12345678910abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "No limit thanks to C++11's variadic templates");
}

TEST(StrCat, VectorBoolReferenceTypes) {
  std::vector<bool> v;
  v.push_back(true);
  v.push_back(false);
  std::vector<bool> const& cv = v;
  // Test that vector<bool>::reference and vector<bool>::const_reference
  // are handled as if the were really bool types and not the proxy types
  // they really are.
  std::string result = ::StrCat(v[0], v[1], cv[0], cv[1]); // NOLINT
  EXPECT_EQ(result, "1010");
}

TEST(StrAppend, CornerCases) {
  std::string result;
  ::StrAppend(&result, "");
  EXPECT_EQ(result, "");
  ::StrAppend(&result, "", "");
  EXPECT_EQ(result, "");
  ::StrAppend(&result, "", "", "");
  EXPECT_EQ(result, "");
  ::StrAppend(&result, "", "", "", "");
  EXPECT_EQ(result, "");
  ::StrAppend(&result, "", "", "", "", "");
  EXPECT_EQ(result, "");
}

TEST(StrAppend, CornerCasesNonEmptyAppend) {
  for (std::string result : {"hello", "a string too long to fit in the SSO"}) {
    const std::string expected = result;
    ::StrAppend(&result, "");
    EXPECT_EQ(result, expected);
    ::StrAppend(&result, "", "");
    EXPECT_EQ(result, expected);
    ::StrAppend(&result, "", "", "");
    EXPECT_EQ(result, expected);
    ::StrAppend(&result, "", "", "", "");
    EXPECT_EQ(result, expected);
    ::StrAppend(&result, "", "", "", "", "");
    EXPECT_EQ(result, expected);
  }
}
